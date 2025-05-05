import json
import os
import re
import torch
from math import ceil
from tqdm import tqdm
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead
from parser import parse_args
from chatgpt import ChatGPT
import openai
from random import shuffle
from pytz import timezone
from datetime import datetime
import logging
import sys

from utils import setup_tokenizer, load_base_model, load_peft_model, prepare_data
from interact import get_conv, get_prompt, run_interaction



# === 하이퍼파라미터 설정 ===
MODEL_NAME = "meta-llama/Meta-Llama-3.1-8B-Instruct"
GAMMA = 0.9
TOTAL_EPOCHS = 2
PPO_EPOCHS = 4
BATCH_SIZE = 4
LEARNING_RATE = 1e-5
INIT_KL_COEF = 0.2
LOG_WITH = 'None'


def create_ppo_trainer(peft_model, tokenizer):

    model = AutoModelForCausalLMWithValueHead(peft_model)
    model.is_peft_model = True

    ppo_config = PPOConfig(
        model_name=MODEL_NAME,
        ppo_epochs=PPO_EPOCHS,
        learning_rate=LEARNING_RATE,
        gamma=GAMMA,
        kl_penalty="kl",
        init_kl_coef=INIT_KL_COEF,
        # target_kl=TARGET_KL,
        # adap_kl_ctrl=ADAPTIVE_KL_CONTROL,
        batch_size=BATCH_SIZE, 
        mini_batch_size=1
    )

    ppo_trainer = PPOTrainer(config=ppo_config, model=model, tokenizer=tokenizer)
    return ppo_trainer


def train(args):
    openai.api_key = args.api_key
    tokenizer = setup_tokenizer(MODEL_NAME)
    base_model = load_base_model(MODEL_NAME)
    model = load_peft_model(base_model, args.model_path)
    model.resize_token_embeddings(len(tokenizer))
    model.config.pad_token_id = tokenizer.pad_token_id

    ppo_trainer = create_ppo_trainer(model, tokenizer)
    rank, world_size = ppo_trainer.accelerator.process_index, ppo_trainer.accelerator.num_processes

    entity2id = json.load(open(os.path.join(args.home, f'data/{args.kg_dataset}/entity2id.json'), 'r'))
    id2entity = {int(v): k for k, v in entity2id.items()}
    chatgpt = ChatGPT(args)

    hit = 0
    avg_turn = 0

    # === 로그 파일 설정 ===
    mdhm = str(datetime.now(timezone('Asia/Seoul')).strftime('%m%d%H%M%S'))
    log_file = os.path.join(args.home, 'results', 'ppo', f'{mdhm}.txt')
    logging.basicConfig(
        level=logging.INFO,  # 출력 레벨 (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        format="%(message)s",
        handlers=[
            logging.FileHandler(log_file, encoding="utf-8"),
            logging.StreamHandler(sys.stdout)  # 콘솔 출력도 포함
        ]
    )
    data_path = os.path.join(args.home, 'data', 'redial_processed_train.json')
    train_data = prepare_data(data_path, rank, world_size, length=5000, is_shuffle=True)
    i = 0
    for epoch in range(TOTAL_EPOCHS):
        
        prompts, responses, rewards, response_masks = [], [], [], []

        while i < len(train_data):
            while len(prompts) < BATCH_SIZE and i < len(train_data):          
                conv_dict = train_data[i]['dialog'].copy()
                target_items = train_data[i]['target_items']

                conv_dict, rec_success, original_conv_len = run_interaction(
                    args, ppo_trainer.model, tokenizer, chatgpt, conv_dict, target_items, entity2id, id2entity
                )
                interaction_num = (len(conv_dict) - original_conv_len) // 2
                i += 1

                if rec_success:
                    hit += 1
                    avg_turn += interaction_num
                # else:
                #     continue
                
                prompt = get_prompt(tokenizer, conv_dict[:original_conv_len])
                prompt_response = prompt
                role_masks = []
                for cidx in range(original_conv_len, len(conv_dict) - 1):
                    is_user = conv_dict[cidx]['role'] == 'user'

                    # 텍스트 확장: 직전까지의 context로 생성
                    response = get_prompt(
                        tokenizer, 
                        conv_dict[:original_conv_len], 
                        conv_dict[original_conv_len:cidx + 1], 
                        add_generation_prompt=is_user
                    )

                    # prompt 이후 새로운 토큰만 추출
                    delta_response = response[len(prompt_response):]
                    prompt_response += delta_response

                    # 이 토큰 길이에 대한 마스크 생성
                    token_len = len(tokenizer(delta_response, add_special_tokens=False).input_ids)
                    role_mask = torch.ones(token_len, dtype=torch.long) if not is_user else torch.zeros(token_len, dtype=torch.long)
                    role_masks.append(role_mask)

                response = prompt_response[len(prompt):]
                reward = 1 if rec_success else -1
                response_mask = torch.cat(role_masks, dim=0).to(dtype=torch.long)

                prompts.append(tokenizer(prompt, return_tensors="pt", add_special_tokens=False)["input_ids"].squeeze(0).to(dtype=torch.long))
                responses.append(tokenizer(response, return_tensors="pt", add_special_tokens=False)["input_ids"].squeeze(0).to(dtype=torch.long))
                rewards.append(torch.tensor([reward], dtype=torch.float32))
                response_masks.append(response_mask)

                logging.info(f"################################# Dialog Case {i} #################################")

                for idx, utt in enumerate(conv_dict):
                    role = utt['role']
                    content = utt['content']
                    logging.info(f"{role}: {content}")
                    if idx == original_conv_len - 1:
                        logging.info("------------------------------------------------------------------------------------")
                logging.info(f"[[[REC_SUCCESS: {rec_success}]]]")
                hit_ratio = hit / (i + 1)
                logging.info(f"[[[hit_ratio: {hit_ratio:.3f}]]]")
                avg_success_turn = avg_turn / hit if hit != 0 else 0
                logging.info(f"[[[avg_success_turn: {avg_success_turn:.3f}]]]")
                logging.info(f"###################################################################################")
                

            ppo_trainer.config.batch_size = len(prompts)
            stats = ppo_trainer.step(prompts, responses, rewards, response_masks)
            # stats = ppo_trainer.step(prompts, responses, rewards)

            logging.info(
                f"Loss(total): {stats['ppo/loss/total']:.4f} | "
                f"Policy Loss: {stats['ppo/loss/policy']:.4f} | "
                f"Value Loss: {stats['ppo/loss/value']:.4f} | "
                f"KL: {stats['objective/kl']:.6f} | "
                f"Entropy: {stats['objective/entropy']:.2f} | "
                f"Mean Reward: {stats['ppo/mean_scores']:.4f} | "
                f"Adv. Mean: {stats['ppo/policy/advantages_mean']:.4f}"
            )
            prompts, responses, rewards, response_masks = [], [], [], []

        if ppo_trainer.accelerator.is_main_process:
            model_path = os.path.join(args.home, 'model_weights', f"ppo_model_{mdhm}_{epoch}")
            ppo_trainer.save_pretrained(model_path)
            logging.info("✅ 모델 저장 완료")

if __name__ == '__main__':
    args = parse_args()
    train(args)
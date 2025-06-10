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
from transformers import get_scheduler

from utils import setup_tokenizer, load_base_model, load_peft_model, prepare_data
from interact import get_prompt, run_interaction



# === 하이퍼파라미터 설정 ===
# MODEL_NAME = "meta-llama/Meta-Llama-3.1-8B-Instruct"
# GAMMA = 0.9
# TOTAL_EPOCHS = 2
# PPO_EPOCHS = 4
# BATCH_SIZE = 4
# LEARNING_RATE = 1e-5
# INIT_KL_COEF = 0.2
# LOG_WITH = 'None'


def create_ppo_trainer(args, peft_model, tokenizer):

    model = AutoModelForCausalLMWithValueHead(peft_model)
    model.is_peft_model = True
    
    # REF
    if args.ref_model:
        ref_model = peft_model.base_model.model  # PEFT 이전 base
        ref_model = AutoModelForCausalLMWithValueHead(ref_model)
        ref_model.is_peft_model = False
        ref_model.eval()
    else:
        ref_model = None
    
    ppo_config = PPOConfig(
        model_name=args.model_name,
        ppo_epochs=args.ppo_epoch,
        learning_rate=args.learning_rate,
        gamma=args.gamma,
        lam=args.lam,
        kl_penalty="kl",
        init_kl_coef=args.init_kl_coef,
        target_kl=args.target_kl,
        adap_kl_ctrl=args.adap_kl_ctrl,
        batch_size=args.batch_size,
        # gradient_accumulation_steps=args.gradient_accumulation_steps,
        mini_batch_size=1
    )
    
    # Optimizer (optional: TRL creates one automatically if omitted)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)

    # Warmup + linear decay scheduler
    lr_scheduler = get_scheduler(
        name="linear",                 # or "cosine", "polynomial", etc.
        optimizer=optimizer,
        num_warmup_steps=100,
        num_training_steps=100000,    # or set to large number or estimate (can be safely overshot)
    )
    ppo_trainer = PPOTrainer(config=ppo_config, model=model, ref_model=ref_model, tokenizer=tokenizer)
    # ppo_trainer = PPOTrainer(config=ppo_config, model=model, tokenizer=tokenizer, optimizer=optimizer, lr_scheduler=lr_scheduler)
    return ppo_trainer


def setup_logger(log_file_path):
    # 기존 핸들러 제거
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    os.makedirs(os.path.dirname(log_file_path), exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[
            logging.FileHandler(log_file_path, encoding="utf-8"),
            logging.StreamHandler(sys.stdout)
        ]
    )


def train(args):
    openai.api_key = args.api_key
    tokenizer = setup_tokenizer(args.model_name)
    base_model = load_base_model(args.model_name)
    model = load_peft_model(base_model, args.model_path)
    model.resize_token_embeddings(len(tokenizer))
    model.config.pad_token_id = tokenizer.pad_token_id

    ppo_trainer = create_ppo_trainer(args, model, tokenizer)
    rank, world_size = ppo_trainer.accelerator.process_index, ppo_trainer.accelerator.num_processes

    entity2id = json.load(open(os.path.join(args.home, f'data/{args.kg_dataset}/entity2id.json'), 'r'))
    id2entity = {int(v): k for k, v in entity2id.items()}
    chatgpt = ChatGPT(args)

    data_path = os.path.join(args.home, 'data', args.train_data)
    mdhm = str(datetime.now(timezone('Asia/Seoul')).strftime('%m%d%H%M%S'))
    log_name = args.log_name
    train_data = prepare_data(data_path, rank, world_size, start=args.start, end=args.end, is_shuffle=True)

    for epoch in range(args.epoch):

        # === 로그 파일 설정 ===
        log_file = os.path.join(args.home, 'results', 'ppo', f'{mdhm}_{log_name}_E{epoch + 1}.txt')
        setup_logger(log_file)
  
        prompts, responses, rewards, response_masks = [], [], [], []
        i = 0 # sample num
        dialog_id = args.start  # dialog id
        hit = 0
        avg_turn = 0
        sample_cnt = 0

        while i < len(train_data): 
            hit_batch = 0
            while len(prompts) < args.batch_size and i < len(train_data):    
                default_conv_dict = train_data[i]['dialog'].copy()
                target_items = train_data[i]['target_items']
                base_turn = train_data[i]['base_turn']

                for _ in range(args.num_explore):
                    conv_dict, rec_success, original_conv_len, _, _, _, _ = run_interaction(
                        args, ppo_trainer.model, tokenizer, chatgpt, default_conv_dict, target_items, 
                        entity2id, id2entity, last_turn_recommend=args.last_turn_recommend, rec_success_recommend=args.rec_success_recommend, is_train=True
                    )

                    interaction_num = (len(conv_dict) - original_conv_len) // 2
                    if interaction_num > args.max_train_turn:
                        original_conv_len += 2 * (interaction_num - args.max_train_turn)

                    prompt = get_prompt(tokenizer, conv_dict[:original_conv_len], few_shot=args.few_shot)
                    prompt_response = prompt
                    role_masks = []
                    for cidx in range(original_conv_len, len(conv_dict) - 1):
                        is_user = conv_dict[cidx]['role'] == 'user'

                        # 텍스트 확장: 직전까지의 context로 생성
                        response = get_prompt(
                            tokenizer, 
                            conv_dict[:original_conv_len], 
                            conv_dict[original_conv_len:cidx + 1], 
                            add_generation_prompt=is_user,
                            few_shot=args.few_shot
                        )

                        # prompt 이후 새로운 토큰만 추출
                        delta_response = response[len(prompt_response):]
                        prompt_response += delta_response

                        # 이 토큰 길이에 대한 마스크 생성
                        token_len = len(tokenizer(delta_response, add_special_tokens=False).input_ids)
                        role_mask = torch.ones(token_len, dtype=torch.long) if not is_user else torch.zeros(token_len, dtype=torch.long)
                        role_masks.append(role_mask)

                    response = prompt_response[len(prompt):]
                    # reward = 1 if rec_success else -1
                    if args.diff_aware:
                        if rec_success:
                            if base_turn > 5 and interaction_num <= 3:
                                reward = 1.3 # if interaction_num <= 2 else 1.1
                            else:
                                reward = 1 
                        else:
                            reward = -1
                            # if base_turn <= 5 and base_turn <= 3:
                            #     reward = -1.3 # if base_turn <= 2 else -1.1
                            # else:
                            #     reward = -1
                    else:
                        reward = args.reward if rec_success else -args.reward
                    
                    response_mask = torch.cat(role_masks, dim=0).to(dtype=torch.long)

                    prompts.append(tokenizer(prompt, return_tensors="pt", add_special_tokens=False)["input_ids"].squeeze(0).to(dtype=torch.long))
                    responses.append(tokenizer(response, return_tensors="pt", add_special_tokens=False)["input_ids"].squeeze(0).to(dtype=torch.long))
                    rewards.append(torch.tensor([reward], dtype=torch.float32))
                    response_masks.append(response_mask)


                    if rec_success:
                        hit += 1
                        hit_batch += 1
                        avg_turn += interaction_num
                    sample_cnt += 1
                    logging.info(f"################################# Dialog Case {dialog_id+1} #################################")

                    for idx, utt in enumerate(conv_dict):
                        role = utt['role']
                        content = utt['content']
                        logging.info(f"{role}: {content}")
                        if idx == original_conv_len - 1:
                            logging.info("------------------------------------------------------------------------------------")
                    logging.info(f"[[[REC_SUCCESS: {rec_success}]]]")
                    hit_cnt = hit
                    hit_ratio = hit / sample_cnt
                    logging.info(f"[[[hit_cnt: {hit_cnt:.3f}]]]")
                    logging.info(f"[[[hit_ratio: {hit_ratio:.3f}]]]")               
                    avg_success_turn = avg_turn / hit if hit != 0 else 0
                    logging.info(f"[[[avg_success_turn: {avg_success_turn:.3f}]]]")
                    logging.info(f"[[[base_turn: {base_turn:.3f} | reward: {reward:.1f}]]]")
                    logging.info(f"###################################################################################")

                i += 1
                dialog_id += 1

            ppo_trainer.config.batch_size = len(prompts)
            stats = ppo_trainer.step(prompts, responses, rewards, response_masks)
            # stats = ppo_trainer.step(prompts, responses, rewards)

            current_lr = ppo_trainer.optimizer.param_groups[0]['lr']
            hit_batch_ratio = hit_batch / len(prompt)
            current_kl_coef = ppo_trainer.kl_ctl.value

            logging.info(
                f"Loss(total): {stats['ppo/loss/total']:.4f} | "
                f"Policy Loss: {stats['ppo/loss/policy']:.4f} | "
                f"Value Loss: {stats['ppo/loss/value']:.4f} | "
                f"KL: {stats['objective/kl']:.6f} | "
                f"Entropy: {stats['objective/entropy']:.2f} | "
                f"Mean Reward: {stats['ppo/mean_scores']:.4f} | "
                f"Adv. Mean: {stats['ppo/policy/advantages_mean']:.4f} | "
                f"LR: {current_lr:.6e} | "
                f"KL Coef: {current_kl_coef:.6f} | "
                f"Hit_batch: {hit_batch_ratio:.3f}"
            )

            del prompts, responses, rewards, response_masks
            torch.cuda.empty_cache()
            prompts, responses, rewards, response_masks = [], [], [], []

        if ppo_trainer.accelerator.is_main_process:
            model_path = os.path.join(args.home, 'model_weights', f"ppo_model_{mdhm}_{log_name}_E{epoch+1}")
            ppo_trainer.save_pretrained(model_path)
            logging.info("✅ 모델 저장 완료")

            if args.ref_model:
                # === ref_model 갱신 ===
                # base_model은 이미 위에서 load_base_model()로 불러온 상태
                # peft_model = PeftModel.from_pretrained(base_model, model_path)  # ⬅ 저장된 LoRA adapter 로드
                peft_model = load_peft_model(base_model, model_path)

                merged_model = peft_model.merge_and_unload()                    # ⬅ adapter 병합
                ref_model = AutoModelForCausalLMWithValueHead(merged_model)    # ⬅ value head 추가
                ref_model.is_peft_model = False
                ref_model.eval()

                ppo_trainer.ref_model = ref_model

                # # === ref_model 갱신 ===
                # merged_model = ppo_trainer.model.merge_and_unload()
                # ref_model = AutoModelForCausalLMWithValueHead(merged_model)
                # ref_model.is_peft_model = False
                # ref_model.eval()
                # ppo_trainer.ref_model = ref_model

if __name__ == '__main__':
    args = parse_args()
    train(args)
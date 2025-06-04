import json
import os
import re
import torch
from math import ceil
from tqdm import tqdm
from trl import GRPOTrainer, GRPOConfig
from parser import parse_args
from chatgpt import ChatGPT
import openai
from random import shuffle
from pytz import timezone
from datetime import datetime
import logging
import sys
from transformers import get_scheduler, AutoModelForCausalLM

from utils import setup_tokenizer, load_base_model, load_peft_model, prepare_data
from interact import get_conv, get_prompt, run_interaction

def create_grpo_trainer(args, peft_model, tokenizer):
    base_model = peft_model.base_model.model
    model = AutoModelForCausalLM.from_pretrained(base_model.config._name_or_path)
    model.resize_token_embeddings(len(tokenizer))
    model.config.pad_token_id = tokenizer.pad_token_id
    model.eval()

    grpo_config = GRPOConfig(
        model_name=args.model_name,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        mini_batch_size=1,
        beta=args.init_kl_coef,
        loss_type="dr_grpo",
        scale_rewards=True,
        disable_dropout=True,
    )

    def reward_fn(samples, **kwargs):
        rewards = []
        for s in samples:
            rewards.append(kwargs['raw_rewards'].get(s, 0.0))
        return rewards

    grpo_trainer = GRPOTrainer(
        config=grpo_config,
        model=model,
        tokenizer=tokenizer,
        reward_funcs=[reward_fn]
    )
    return grpo_trainer

def setup_logger(log_file_path):
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
    grpo_trainer = create_grpo_trainer(args, model, tokenizer)
    rank, world_size = grpo_trainer.accelerator.process_index, grpo_trainer.accelerator.num_processes

    entity2id = json.load(open(os.path.join(args.home, f'data/{args.kg_dataset}/entity2id.json'), 'r'))
    id2entity = {int(v): k for k, v in entity2id.items()}
    chatgpt = ChatGPT(args)

    data_path = os.path.join(args.home, 'data', args.train_data)
    mdhm = str(datetime.now(timezone('Asia/Seoul')).strftime('%m%d%H%M%S'))
    log_name = args.log_name
    train_data = prepare_data(data_path, rank, world_size, start=args.start, end=args.end, is_shuffle=True)

    for epoch in range(args.epoch):
        log_file = os.path.join(args.home, 'results', 'grpo', f'{mdhm}_{log_name}_E{epoch + 1}.txt')
        setup_logger(log_file)

        samples = []
        raw_rewards = {}
        i = 0
        dialog_id = args.start
        hit = 0
        avg_turn = 0

        while i < len(train_data):
            conv_dict = train_data[i]['dialog'].copy()
            target_items = train_data[i]['target_items']
            base_turn = train_data[i]['base_turn']

            conv_dict, rec_success, original_conv_len = run_interaction(
                args, grpo_trainer.model, tokenizer, chatgpt, conv_dict, target_items, entity2id, id2entity, last_turn_recommend=args.last_turn_recommend, is_train=True
            )
            interaction_num = (len(conv_dict) - original_conv_len) // 2
            if interaction_num > args.max_train_turn:
                original_conv_len += 2 * (interaction_num - args.max_train_turn)

            i += 1
            dialog_id += 1

            if rec_success:
                hit += 1
                avg_turn += interaction_num

            prompt = get_prompt(tokenizer, conv_dict[:original_conv_len], few_shot=args.few_shot)
            full_response = get_prompt(tokenizer, conv_dict[:original_conv_len], conv_dict[original_conv_len:], few_shot=args.few_shot)
            response = full_response[len(prompt):]

            reward = 1 if rec_success else -1
            raw_rewards[response] = reward
            samples.append((prompt, response))

            logging.info(f"==== Dialog {dialog_id} | Success: {rec_success} | Reward: {reward} ====")

        prompts = [tokenizer(p, return_tensors="pt", add_special_tokens=False)["input_ids"].squeeze(0) for p, _ in samples]
        completions = [r for _, r in samples]
        stats = grpo_trainer.step(prompts=prompts, completions=completions, raw_rewards=raw_rewards)

        current_lr = grpo_trainer.optimizer.param_groups[0]['lr']
        logging.info(f"GRPO Step | Loss: {stats['loss']:.4f} | KL: {stats['kl']:.6f} | Entropy: {stats['entropy']:.4f} | LR: {current_lr:.6e}")

        if grpo_trainer.accelerator.is_main_process:
            model_path = os.path.join(args.home, 'model_weights', f"grpo_model_{mdhm}_{log_name}_E{epoch+1}")
            grpo_trainer.save_pretrained(model_path)
            logging.info("✅ 모델 저장 완료")

if __name__ == '__main__':
    args = parse_args()
    train(args)

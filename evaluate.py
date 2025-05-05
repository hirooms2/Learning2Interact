import json
import os
import torch
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    DataCollatorForSeq2Seq
)
from peft import LoraConfig, get_peft_model, TaskType
from parser import parse_args
from datetime import datetime
from pytz import timezone
import logging
import openai
import sys
from random import shuffle
from utils import setup_tokenizer, load_base_model, load_peft_model, prepare_data
from torch.utils.data import Dataset, DataLoader
from chatgpt import ChatGPT
from interact import get_conv, get_prompt, run_interaction
import random

random.seed(2025)

# === Hyperparameters ===
MODEL_NAME = "meta-llama/Meta-Llama-3.1-8B-Instruct"
TOTAL_EPOCHS = 5
LEARNING_RATE = 3e-5
BATCH_SIZE = 4
LOGGING_STEPS = 100

logging.basicConfig(
    level=logging.INFO,  # INFO 이상의 로그 출력
    format='[%(asctime)s] [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

def main(args):
    openai.api_key = args.api_key
    tokenizer = setup_tokenizer(MODEL_NAME)
    model = load_base_model(MODEL_NAME)
    if args.model_path:
        model = load_peft_model(model, args.model_path)
    model.resize_token_embeddings(len(tokenizer))
    model.config.pad_token_id = tokenizer.pad_token_id

    mdhm = str(datetime.now(timezone('Asia/Seoul')).strftime('%m%d%H%M%S'))
    log_file = open(os.path.join(args.home, 'results', 'eval', f'{mdhm}.txt'), 'a', buffering=1, encoding='utf-8')
    
    rank, world_size = 0, 1
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        rank = torch.distributed.get_rank()
        world_size = torch.distributed.get_world_size()

    # Prepare dataset
    test_dataset = prepare_data(
        '/home/user/junpyo/Learning2Interact/data/redial_processed_test.json', rank, world_size, is_shuffle=True, length=1000
    )

    entity2id = json.load(open(os.path.join(args.home, f'data/{args.kg_dataset}/entity2id.json'), 'r'))
    id2entity = {int(v): k for k, v in entity2id.items()}
    chatgpt = ChatGPT(args)
    hit = 0
    avg_turn = 0
    all_samples = []
    for i in tqdm(range(len(test_dataset))):
        conv_dict = test_dataset[i]['dialog'].copy()
        target_items = test_dataset[i]['target_items']
        conv_dict, rec_success, original_conv_len = run_interaction(
            args, model, tokenizer, chatgpt, conv_dict, target_items, entity2id, id2entity, last_turn_recommed=False
        )
        interaction_num = (len(conv_dict) - original_conv_len) // 2
        all_samples.append({'context': conv_dict, 'original_conv_len': original_conv_len})
        # print(tokenizer.apply_chat_template(conv_dict, tokenize=False, add_generation_prompt=False))
        # print()
        result_str = f'# Case: {i+1}\n'
        for idx, utt in enumerate(conv_dict):
            if idx == original_conv_len:
                result_str += '-------------------------------------\n'
            result_str += f"{utt['role']}: {utt['content']}\n"


        log_file.write(result_str + '\n\n')
        if rec_success:
            hit += 1
            avg_turn += interaction_num

    hit_ratio = hit / len(test_dataset)
    avg_success_turn = avg_turn / hit if hit != 0 else 0
    logging.info(f"Hit_ratio: {hit_ratio:.3f}")
    logging.info(f"Avg_success_turn: {avg_success_turn:.3f}")


if __name__ == "__main__":
    args = parse_args()
    main(args)

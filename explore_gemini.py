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
#import openai
import sys
from random import shuffle
from utils import setup_tokenizer, load_base_model, load_peft_model, prepare_data
from torch.utils.data import Dataset, DataLoader
#from chatgpt import ChatGPT
from chatgpt_gemini import ChatGPT
import google.generativeai as genai

from interact_gemini import run_explore, run_explore_gpt
import random


# === Hyperparameters ===
# MODEL_NAME = "meta-llama/Meta-Llama-3.1-8B-Instruct"
# TOTAL_EPOCHS = 5
# LEARNING_RATE = 3e-5
# BATCH_SIZE = 4
# LOGGING_STEPS = 100


def main(args):
    #openai.api_key = args.api_key
    genai.configure(api_key=args.gemini_api_key)

    mdhm = str(datetime.now(timezone('Asia/Seoul')).strftime('%m%d%H%M%S'))
    # log_file = os.path.join(args.home, 'results', 'eval', f'{mdhm}.txt')
    dir_path = os.path.join(args.home, 'results', 'explore')
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    log_file = os.path.join(dir_path, f'{mdhm}_{args.log_name}.txt')
    json_path = os.path.join(dir_path, f'{mdhm}_{args.log_name}.json')
    json_file = open(json_path, 'a', buffering=1, encoding='UTF-8')
    
    logging.basicConfig(
        level=logging.INFO,  # 출력 레벨 (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        format="%(message)s",
        handlers=[
            logging.FileHandler(log_file, encoding="utf-8"),
            logging.StreamHandler(sys.stdout)  # 콘솔 출력도 포함
        ]
    )

    rank, world_size = 0, 1
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        rank = torch.distributed.get_rank()
        world_size = torch.distributed.get_world_size()

    # Prepare dataset
    data_path = os.path.join(args.home, 'data', args.test_data)
    train_dataset = prepare_data(
        data_path, rank, world_size, start=args.start, end=args.end, is_shuffle=False
    )

    entity2id = json.load(open(os.path.join(args.home, f'data/{args.kg_dataset}/entity2id.json'), 'r'))
    id2entity = {int(v): k for k, v in entity2id.items()}
    chatgpt = ChatGPT(args)
    dialog_id = 0
    hit = 0
    avg_turn = 0
    all_samples = []
    for i in tqdm(range(len(train_dataset))):
        dialog_id += 1
        conv_dict = train_dataset[i]['dialog'].copy()
        target_items = train_dataset[i]['target_items']

        # TH: is_train False로
        conv_dict, rec_success, original_conv_len, rec_names, rec_ids, topk_names, topk_ids = run_explore_gpt(
            args, chatgpt, conv_dict, target_items, entity2id, id2entity, last_turn_recommend=True, is_train=False
        )
        interaction_num = (len(conv_dict) - original_conv_len) // 2
        all_samples.append({'context': conv_dict, 'original_conv_len': original_conv_len})

        if rec_success:
            hit += 1
            avg_turn += interaction_num

        ## Logging Dictionary
        output = {
                'id': dialog_id,
                'given_dialog': [],
                'generated_dialog': [],
                'rec_success' : None,
                'hit_ratio' : None,
                'rec_names' : None,
                'rec_ids' : None,
                'topk_names' : None,
                'topk_ids' : None
            }

        logging.info(f"################################# Dialog Case {i} #################################")

        for idx, utt in enumerate(conv_dict):
            role = utt['role']
            content = utt['content']
            role_content = {'role': role, 'content': content}
            if idx < original_conv_len:
                output['given_dialog'].append(role_content)
            else:
                output['generated_dialog'].append(role_content)
            logging.info(f"{role}: {content}")
            if idx == original_conv_len - 1:
                logging.info("------------------------------------------------------------------------------------")
        logging.info(f"[[[REC_SUCCESS: {rec_success}]]]")
        hit_cnt = hit
        hit_ratio = hit / (i + 1)
        logging.info(f"[[[hit_cnt: {hit_cnt:.3f}]]]")
        logging.info(f"[[[hit_ratio: {hit_ratio:.3f}]]]")
        avg_success_turn = avg_turn / hit if hit != 0 else 0
        logging.info(f"[[[avg_success_turn: {avg_success_turn:.3f}]]]")
        logging.info(f"[[[rec_names: {rec_names}]]]")
        logging.info(f"[[[rec_ids: {rec_ids}]]]")
        logging.info(f"[[[topk_names: {topk_names}]]]")
        logging.info(f"[[[topk_ids: {topk_ids}]]]")        
        
        logging.info(f"###################################################################################")

        output['rec_success'] = rec_success
        output['hit_ratio'] = f"{hit_ratio:.3f}"
        output['rec_names'] = rec_names
        output['rec_ids'] = rec_ids
        output['topk_names'] = topk_names
        output['topk_ids'] = topk_ids

        json_file.write(
            json.dumps(output, ensure_ascii=False) + "\n")

    hit_ratio = hit / len(train_dataset)
    avg_success_turn = avg_turn / hit if hit != 0 else 0
    logging.info(f"Hit_ratio: {hit_ratio:.3f}")
    logging.info(f"Avg_success_turn: {avg_success_turn:.3f}")


if __name__ == "__main__":
    args = parse_args()
    main(args)

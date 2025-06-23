import json
import os
import re
import torch
from math import ceil
from tqdm import tqdm
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import get_peft_model, LoraConfig, TaskType
from parser import parse_args
from interact import get_conv, get_prompt
from chatgpt import ChatGPT
import nltk
from thefuzz import fuzz
import openai
from random import shuffle
from pytz import timezone
from datetime import datetime
import logging
import sys
from peft import PeftModel



def prepare_data(data_path, rank, world_size, start=0, end=0, is_shuffle=True):
    all_data = json.load(open(data_path, 'r', encoding='utf-8'))
    # start = 1191
    if end:
        all_data = all_data[start:end]
    elif start:
        all_data = all_data[start:]
        
    if is_shuffle:
        shuffle(all_data)

    return all_data[rank::world_size]


def load_base_model(model_name):
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )

    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )

    return base_model


def load_peft_model(model, model_path, is_trainable=True):
    if model_path != '':
        peft_model = PeftModel.from_pretrained(model, model_path, is_trainable=is_trainable)
    else:
        lora_config = LoraConfig(
            r=64,
            lora_alpha=16,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        peft_model = get_peft_model(model, lora_config)
    peft_model.print_trainable_parameters()

    return peft_model


def setup_tokenizer(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        print('Set a pad token as <|pad|> in the tokenizer')
        tokenizer.add_special_tokens({"pad_token": "<|pad|>"})
    tokenizer.padding_side = "left"
    return tokenizer
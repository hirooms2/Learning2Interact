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
import sys
from random import shuffle

# === Hyperparameters ===
# MODEL_NAME = "meta-llama/Meta-Llama-3.1-8B-Instruct"
# TOTAL_EPOCHS = 5
# LEARNING_RATE = 3e-5
# BATCH_SIZE = 4
# LOGGING_STEPS = 100

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

def setup_tokenizer(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    tokenizer.add_special_tokens({"pad_token": "<|pad|>"})
    tokenizer.padding_side = "right"
    return tokenizer

def prepare_dataset(data_path, tokenizer, rank, world_size):
    all_data = json.load(open(data_path, 'r', encoding='utf-8'))
    # all_data = all_data[:1000]
    shuffle(all_data)

    # 데이터 분산 처리
    data = all_data[rank::world_size]

    dataset = []
    
    for example in data:
        if not example['success']:
            continue
        
        context = example['dialog'][-2:]
        context_len = len(tokenizer.apply_chat_template(context, tokenize=True, add_generation_prompt=False))
        interaction = [{"role": "assistant", "content": i['content'].split("\n11: ")[0]} if "I would recommend the following items: " in i['content'] else i for i in example['interaction'][:-1]]
        context_interaction = context + interaction
        
        prompt = tokenizer.apply_chat_template(context_interaction, tokenize=False, add_generation_prompt=False)
        if prompt in dataset:
            continue
        tokenized_prompt = tokenizer(prompt, truncation=True, max_length=512, add_special_tokens=False)
        input_ids = tokenized_prompt.input_ids
        labels = input_ids.copy()
        # labels = [token if idx >= context_len else -100 for idx, token in enumerate(input_ids)]
        
        dataset.append({'input_ids':input_ids, "labels": labels})
        
    return dataset

def main(args):
    tokenizer = setup_tokenizer(args.model_name)
    base_model = load_base_model(args.model_name)
    base_model.resize_token_embeddings(len(tokenizer))
    base_model.config.pad_token_id = tokenizer.pad_token_id

    # LoRA Configuration
    lora_config = LoraConfig(
        r=64,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )

    model = get_peft_model(base_model, lora_config)

    rank, world_size = 0, 1
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        rank = torch.distributed.get_rank()
        world_size = torch.distributed.get_world_size()

    # Prepare dataset
    data_path = os.path.join(args.home, 'data', args.train_data)
    tokenized_dataset = prepare_dataset(
        data_path, tokenizer, rank, world_size
    )

    # Logging 설정
    mdhm = datetime.now(timezone('Asia/Seoul')).strftime('%m%d%H%M%S')
    log_file = os.path.join(args.home, 'results', 'sft', f'{mdhm}.txt')
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[
            logging.FileHandler(log_file, encoding="utf-8"),
            logging.StreamHandler(sys.stdout)
        ]
    )
    # mdhm = datetime.now(timezone('Asia/Seoul')).strftime('%m%d%H%M%S')
    model_path = os.path.join(args.home, 'model_weights', f"sft_model_{mdhm}")

    training_args = TrainingArguments(
        output_dir=model_path,
        num_train_epochs=args.epoch,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=4,
        learning_rate=args.learning_rate,
        logging_steps=args.logging_steps,
        save_strategy='no',
        bf16=True,
        remove_unused_columns=False,
        # logging_dir="./logs",
        # report_to="wandb" if args.use_wandb else "none",
    )

    # data_collator = DataCollatorForSeq2Seq(
    #     model=model,
    #     tokenizer=tokenizer,
    # )

    data_collator = DataCollatorForSeq2Seq(
        tokenizer, 
        pad_to_multiple_of=8, 
        return_tensors="pt",
        padding=True)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    # 학습 시작
    logging.info("🚀 SFT 학습 시작")
    trainer.train()

    # 모델 저장
    # LoRA adapter 저장
    # model.save_pretrained(model_path)
    # tokenizer.save_pretrained(model_path)    
    # logging.info("✅ 모델 저장 완료")
    # 모델 merge 및 저장 (LoRA → base weights에 합치기)
    merged_model = model.merge_and_unload()
    merged_model.save_pretrained(model_path + "_merged")
    tokenizer.save_pretrained(model_path + "_merged")
    logging.info("✅ Merge된 모델 저장 완료")


if __name__ == "__main__":
    args = parse_args()
    main(args)

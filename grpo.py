"""
End‑to‑end GRPO fine‑tuning script
=================================
* replaces PPOTrainer ➜ GRPOTrainer (our custom fork in `grpo_trainer.py`)
* normalises rewards across multiple explorations (`num_explore`) **per dialog**
* keeps LoRA / PEFT, reference‑model merge etc. from original script

Assumes neighbouring files:
  ▸ grpo_trainer.py   – our modified trainer (shown previously)
  ▸ utils.py          – tokenizer / model utilities
  ▸ interact.py       – run_interaction(), get_prompt() helpers  
  ▸ chatgpt.py        – OpenAI wrapper
  ▸ parser.py         – argparse definitions

CLI (example)
-------------
```bash
python train_grpo.py \
  --model_name meta-llama/Meta-Llama-3.1-8B-Instruct \
  --model_path ./lora_ckpt \
  --train_data train.json \
  --epoch 2 --batch_size 8 --num_explore 4 \
  --api_key $OPENAI_KEY
```
"""

import json, os, sys, re, logging
from datetime import datetime
from math import ceil
from random import shuffle
from typing import List, Tuple

import torch
from torch import Tensor
from torch.optim import AdamW
from transformers import get_scheduler
from tqdm import tqdm
from pytz import timezone

from trl import AutoModelForCausalLMWithValueHead, PPOConfig
from mytrl.grpo_trainer import GRPOTrainer, GRPOConfig          # ← our fork

# local helpers -------------------------------------------------------------
from utils     import setup_tokenizer, load_base_model, load_peft_model, prepare_data
from interact  import get_prompt, run_interaction, instruction
from chatgpt   import ChatGPT
from parser    import parse_args

# ---------------------------------------------------------------------------
# Logger helper
# ---------------------------------------------------------------------------

def setup_logger(log_file: str):
    for h in logging.root.handlers[:]:
        logging.root.removeHandler(h)
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    logging.basicConfig(
        level   = logging.INFO,
        format  = "%(message)s",
        handlers=[logging.FileHandler(log_file, encoding="utf-8"),
                  logging.StreamHandler(sys.stdout)],
    )

# ---------------------------------------------------------------------------
# GRPO trainer factory
# ---------------------------------------------------------------------------

def create_grpo_trainer(args):
    """Instantiate model ➜ GRPOTrainer (+ optional ref‑model)."""
    tokenizer = setup_tokenizer(args.model_name)

    # ---- active (PEFT) model ------------------------------------------------
    base_model = load_base_model(args.model_name)
    base_model.resize_token_embeddings(len(tokenizer))
    base_model.config.pad_token_id = tokenizer.pad_token_id

    peft_model = load_peft_model(base_model, args.model_path)
    model      = AutoModelForCausalLMWithValueHead(peft_model)
    model.is_peft_model = True

    # ---- reference model (optional) ----------------------------------------
    ref_model = None
    if args.ref_model:
        
        ref_path = os.path.join(args.home, 'model_weights', args.ref_path) if args.ref_path else args.model_path

        ref_base = load_base_model(args.model_name)
        ref_base.resize_token_embeddings(len(tokenizer))
        ref_base.config.pad_token_id = tokenizer.pad_token_id
        ref_base  = load_peft_model(ref_base, ref_path, is_trainable=False)
        ref_model = AutoModelForCausalLMWithValueHead(ref_base)
        ref_model.is_peft_model = False
        ref_model.eval()

    # ---- config -------------------------------------------------------------
    args.scale_rewards = False if args.loss_type == 'dr_grpo' else True
    cfg = GRPOConfig(
        model_name      = args.model_name,
        ppo_epochs      = args.ppo_epoch,
        learning_rate   = args.learning_rate,
        gamma           = 1.0,        # GRPO – value head not used
        lam             = 0.0,
        vf_coef         = 0.0,        # disable value‑loss
        kl_penalty      = "kl",
        beta            = args.init_kl_coef,
        init_kl_coef    = args.init_kl_coef,
        target_kl       = args.target_kl,
        batch_size      = args.batch_size,
        loss_type       = args.loss_type,
        adap_kl_ctrl    = args.adap_kl_ctrl,
        mini_batch_size = 1,
    )

    # ---- trainer ------------------------------------------------------------
    trainer = GRPOTrainer(config=cfg, model=model, ref_model=ref_model, tokenizer=tokenizer)
    return trainer, model, tokenizer

# ---------------------------------------------------------------------------
# util: build response string + per‑token role‑mask
# ---------------------------------------------------------------------------

def build_response_and_mask(tokenizer, conv: list, orig_len: int, *, few_shot: bool=False):
    """Return (response_text, mask_tensor); mask=1 for **assistant** tokens, 0 for user tokens."""
    prompt_response = get_prompt(tokenizer, conv[:orig_len], few_shot=few_shot)
    role_masks: List[Tensor] = []

    for idx in range(orig_len, len(conv) - 1):
        is_user = conv[idx]['role'] == 'user'
        resp = get_prompt(tokenizer,
                          conv[:orig_len], conv[orig_len:idx + 1],
                          add_generation_prompt=is_user, few_shot=few_shot)
        delta = resp[len(prompt_response):]
        prompt_response += delta
        tok_len  = len(tokenizer(delta, add_special_tokens=False).input_ids)
        role_mask = torch.zeros(tok_len, dtype=torch.long) if is_user else torch.ones(tok_len, dtype=torch.long)
        role_masks.append(role_mask)

    response_text = prompt_response[len(get_prompt(tokenizer, conv[:orig_len], few_shot=few_shot)):]
    return response_text, torch.cat(role_masks, dim=0)

# ---------------------------------------------------------------------------
# Print roll-out dialog
# ---------------------------------------------------------------------------

def print_dialog(dialog_id, conv_dict, original_conv_len, rec_success, hit, sample_cnt, avg_turn, base_turn, reward, iter, num_generations):
    logging.info(f"################################# Dialog Case {dialog_id+1} ({iter} / {num_generations}) #################################")

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
    
# ---------------------------------------------------------------------------
# main training loop
# ---------------------------------------------------------------------------

def train(args):
    # ---- external services --------------------------------------------------
    import openai ; openai.api_key = args.api_key

    trainer, model, tokenizer = create_grpo_trainer(args)
    rank, world = trainer.accelerator.process_index, trainer.accelerator.num_processes

    # ---- data & helpers -----------------------------------------------------
    entity2id = json.load(open(os.path.join(args.home, f"data/{args.kg_dataset}/entity2id.json")))
    id2entity = {int(v): k for k, v in entity2id.items()}
    chatgpt   = ChatGPT(args)
    data_path = os.path.join(args.home, 'data', args.train_data)
    train_data = prepare_data(data_path, rank, world, start=args.start, end=args.end, is_shuffle=True)

    # ---- logging ------------------------------------------------------------
    tag      = datetime.now(timezone('Asia/Seoul')).strftime('%m%d%H%M%S')
    log_name = args.log_name

    # ---- bookkeeping --------------------------------------------------------
    hit, success_turn_sum, seen = 0, 0, 0

    # ------------------------------------------------------------------------
    for epoch in range(args.epoch):
        log_file = os.path.join(args.home, 'results', 'grpo', f'{tag}_{log_name}_E{epoch+1}.txt')
        setup_logger(log_file)

        prompts, responses, rewards, masks = [], [], [], []
        # sft_prompts, sft_responses = [], []
        # sft_max_len = 0

        sample_idx = args.resume_start if args.resume_start != 0 and epoch == 0 else 0
        step_num = 1
        while sample_idx < len(train_data):
            # ───────────────────────────────── group (one dialog) ──────────────
            record_buf: List[Tuple[str,str,Tensor,float]] = []
            sample = train_data[sample_idx]

            # ───────────────────────────────── Data prepare for SFT ──────────────
            if args.off_policy:
                sft_conv_dict = sample['dialog'] + sample['interaction']
                sft_orig_len = len(sample['dialog'])
                sft_prompt = get_prompt(tokenizer, sft_conv_dict[:sft_orig_len], few_shot=args.few_shot)
                sft_response, sft_role_mask = build_response_and_mask(tokenizer, sft_conv_dict, sft_orig_len, few_shot=args.few_shot)
                record_buf.append((sft_prompt, sft_response, sft_role_mask, 1.0))

            target_turn_num = args.turn_num - args.turn_num_offset

            for g_idx in range(args.num_generations):
                conv_dict, rec_success, orig_len, rec_names, rec_ids, topk_names, topk_ids = run_interaction(
                    args, trainer.model, tokenizer, chatgpt,
                    sample['dialog'].copy(), sample['target_items'],
                    entity2id, id2entity,
                    last_turn_recommend=args.last_turn_recommend,
                    rec_success_recommend=args.rec_success_recommend,
                    is_train=True,
                )
                interaction_num = (len(conv_dict) - orig_len) // 2
                if interaction_num > target_turn_num:
                    conv_dict = conv_dict[:-2 * (interaction_num-target_turn_num)]

                prompt  = get_prompt(tokenizer, conv_dict[:orig_len], few_shot=args.few_shot)
                response, role_mask = build_response_and_mask(tokenizer, conv_dict, orig_len, few_shot=args.few_shot)

                raw_reward = 1.0 if rec_success and interaction_num <= target_turn_num else -args.reward

                if sample['base_turn'] > 3:
                    if interaction_num < args.turn_num:
                        raw_reward += args.bonus
                else:
                    if interaction_num < sample['base_turn']:
                        raw_reward += args.bonus

                record_buf.append((prompt, response, role_mask, raw_reward))

                # stats
                seen += 1
                if rec_success:
                    hit += 1
                    success_turn_sum += interaction_num

                # Print roll-out dialog
                print_dialog(sample_idx, conv_dict, orig_len, rec_success, hit, seen, success_turn_sum, sample['base_turn'], raw_reward, g_idx+1, args.num_generations)

            sample_idx += 1

            # ---- reward normalisation (GRPO) ---------------------------------
            raw_r = torch.tensor([r for *_, r in record_buf], dtype=torch.float32)
            mu    = raw_r.mean()
            sigma = raw_r.std() if args.scale_rewards else 1.0
            norm_r = (raw_r - mu) / (sigma + 1e-6)

            # discard samples that have equal rewards: torch.isclose(raw_r.std(), torch.tensor(0.0))
            if torch.isclose(raw_r.std(), torch.tensor(0.0)) and args.dynamic_sampling:
                print("Drop the samples")
                continue
            else:
                # ---- push to trainer buffers -------------------------------------
                for (prompt, response, mask_t, _), r in zip(record_buf, norm_r):
                    prompts.append(tokenizer(prompt, return_tensors="pt", add_special_tokens=False).input_ids.squeeze(0))
                    responses.append(tokenizer(response, return_tensors="pt", add_special_tokens=False).input_ids.squeeze(0))
                    rewards.append(r.unsqueeze(0))
                    masks.append(mask_t)

            # reach batch
            if len(prompts) >= args.batch_size or sample_idx == len(train_data):
                trainer.config.batch_size = len(prompts)
                
                stats = trainer.step(prompts, responses, rewards, masks)

                # ---- simple logging ---------------------------------------
                logging.info(
                    f"Step: {step_num} | loss: {stats['ppo/loss/policy']:.4f} | "
                    f"kl: {stats['objective/kl']:.3f} | kl_coef: {trainer.kl_ctl.value:.4f}" )
                prompts, responses, rewards, masks = [], [], [], []

                step_num+=1
                torch.cuda.empty_cache()

            # ---- save -------------------------------------------------
            if trainer.accelerator.is_main_process and (sample_idx % args.save_turn == 0 or sample_idx == len(train_data)):
                out_dir = os.path.join(args.home, 'model_weights', f"grpo_{tag}_{log_name}_E{epoch+1}_S{sample_idx}")
                trainer.save_pretrained(out_dir)
                logging.info(f"✅ saved to {out_dir}")

    # final stats
    if trainer.accelerator.is_main_process:
        logging.info("=================== training finished ===================")
        logging.info(f"hit_ratio  : {hit/seen:.4f}")
        logging.info(f"avg_turn   : {success_turn_sum/max(hit,1):.2f}")

# ---------------------------------------------------------------------------
if __name__ == '__main__':
    args = parse_args()    
    train(args)

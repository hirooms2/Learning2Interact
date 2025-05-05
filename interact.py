
from transformers import GenerationConfig
import torch
import os
import json
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm


instruction = """You are a recommender engaging in a conversation with the user to provide recommendations.
You must follow the instructions below during the chat:

1. If you have sufficient confidence in the user's preferences, you should recommend 10 items the user is most likely to prefer without any explanations. The recommendation list can contain items that were already mentioned in the dialog. The format of the recommendation list is: no. title (year).
2. If you do not have sufficient confidence in the user's preferences, you should ask the user about their preferences.

You must either recommend or ask about the user's preferences; you must not do both simultaneously."""


def get_prompt(tokenizer, context_list, interaction_list: list = [], add_generation_prompt: bool = True):
    
    context = context_list[-5:]
    context.insert(0, {'role': 'system', 'content': instruction})
    context = context + interaction_list
    full_prompt = tokenizer.apply_chat_template(context, tokenize=False, add_generation_prompt=add_generation_prompt)
    
    return full_prompt


def get_conv(args, model, tokenizer, context):
    full_prompt = get_prompt(tokenizer, context)
    model.eval()

    inputs = tokenizer(full_prompt, return_tensors="pt").to("cuda")
        
    input_ids = inputs["input_ids"].to("cuda")
    attention_mask = inputs["attention_mask"].to("cuda")

    generation_config = GenerationConfig(
            num_beams=args.num_beams,
            num_return_sequences=args.num_beams,
            return_dict_in_generate=True,
        )

    with torch.no_grad():
        generation_output = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            generation_config=generation_config,
            return_dict_in_generate=True,
            max_new_tokens=args.max_new_tokens,
            pad_token_id=tokenizer.pad_token_id
        )
    s = generation_output.sequences
    output = tokenizer.batch_decode(s, skip_special_tokens=True)[0]
    
    generated_responses = output[output.rfind('assistant\n'):].split('assistant\n')[-1].replace('\n','').strip()

    return generated_responses


def run_interaction(args, model, tokenizer, chatgpt, conv_dict, target_items, entity2id, id2entity, last_turn_recommed=False, rec_success_recommend=False):
    original_conv_len = len(conv_dict)
    goal_item_str = ', '.join([f'"{item}"' for item in target_items])
    seeker_prompt = chatgpt.get_instruction(goal_item_str)
    for utt in conv_dict:
        seeker_prompt += f"{'Recommender' if utt['role'] == 'assistant' else 'Seeker'}: {utt['content']}\n"

    rec_success = False
    for t in range(args.turn_num):
        recommender_text = get_conv(args, model, tokenizer, conv_dict)
        rec_items = chatgpt.get_rec(conv_dict, recommender_text)
        rec_labels = [entity2id[item] for item in target_items]

        rec_success = any(rec_label in rec_items[0][:args.topk] for rec_label in rec_labels)

        if t == args.turn_num - 1 and last_turn_recommed:
            rec_items_str = "".join(f"{j+1}: {id2entity[rec]}" for j, rec in enumerate(rec_items[0][:10]))
            recommender_text = f"With that in mind, here are some recommendations: {rec_items_str}"

        seeker_prompt += f"Recommender: {recommender_text}\nSeeker:"
        seeker_full_response = chatgpt.annotate_completion(seeker_prompt).strip()
        crs_intent = seeker_full_response.split('2. Response:')[0].strip()
        seeker_text = seeker_full_response.split('2. Response:')[-1].split('Response:')[-1].strip()
        rec_success = rec_success and 'inquiry' not in crs_intent.lower()

        conv_dict += [{"role": "assistant", "content": recommender_text},
                      {"role": "user", "content": seeker_text}]
        seeker_prompt += f"{seeker_text}\n"

        if rec_success:
            break

    return conv_dict, rec_success, original_conv_len
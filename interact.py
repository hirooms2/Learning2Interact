
from transformers import GenerationConfig
import torch
import os
import json
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
import re
import nltk
from thefuzz import fuzz


instruction = """You are a recommender engaging in a conversation with the user to provide recommendations.
You must follow the instructions below during the chat:

1. If you have sufficient confidence in the user's preferences, you should recommend 10 items the user is most likely to prefer without any explanations. The recommendation list can contain items that were already mentioned in the dialog. The format of the recommendation list is: no. title (year).
2. If you do not have sufficient confidence in the user's preferences, you should ask the user about their preferences.

You must either recommend or ask about the user's preferences; you must not do both simultaneously."""

year_pattern = re.compile(r'\(\d+\)')


def get_prompt(tokenizer, context_list, interaction_list: list = [], add_generation_prompt: bool = True):
    
    context = context_list[-5:]
    context.insert(0, {'role': 'system', 'content': instruction})
    context = context + interaction_list
    full_prompt = tokenizer.apply_chat_template(context, tokenize=False, add_generation_prompt=add_generation_prompt)
    
    return full_prompt


def get_conv(args, model, tokenizer, context):
    full_prompt = get_prompt(tokenizer, context)
    model.eval()

    inputs = tokenizer(full_prompt, return_tensors="pt")
        
    input_ids = inputs["input_ids"].to("cuda")
    attention_mask = inputs["attention_mask"].to("cuda")
    
    generation_config = GenerationConfig(
        num_beams=args.num_beams,
        num_return_sequences=args.num_beams,
        return_dict_in_generate=True,
    )
    if args.do_sample:
        generation_config.do_sample = True
        generation_config.temperature = args.temperature
        generation_config.top_p = args.top_p
        generation_config.top_k = args.top_k

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
    outputs = tokenizer.batch_decode(s, skip_special_tokens=True)
    generated_responses = [output[output.rfind('assistant\n'):].split('assistant\n')[-1].replace('\n','').strip() for output in outputs]
    # generated_responses = output[output.rfind('assistant\n'):].split('assistant\n')[-1].replace('\n','').strip()

    return generated_responses


def run_interaction(args, model, tokenizer, chatgpt, conv_dict, target_items, entity2id, id2entity, last_turn_recommed=False, rec_success_recommend=False, is_train=True):
    original_conv_len = len(conv_dict)
    goal_item_str = ', '.join([f'"{item}"' for item in target_items])
    seeker_prompt = chatgpt.get_instruction(goal_item_str)
    for utt in conv_dict:
        seeker_prompt += f"{'Recommender' if utt['role'] == 'assistant' else 'Seeker'}: {utt['content']}\n"

    rec_success = False
    for t in range(args.turn_num):
        recommender_texts = get_conv(args, model, tokenizer, conv_dict)
        rec_labels = [entity2id[item] for item in target_items]

        # TH: beam이 1보다 클 경우, 응답을 하나씩 조사하여 정답을 맞춘 것을 최종 추천 응답으로 설정 (학습 시에만 해당) (최선을 다해서 추천을 해보게 함)
        # if not is_train:
        recommender_texts = [recommender_texts[0]]

        for rtxt_idx in range(len(recommender_texts)):
            recommender_text = recommender_texts[rtxt_idx]
            rec_items = chatgpt.get_rec(conv_dict, recommender_text)
            ## 수정 by BS
            rec_success = any(rec_label in rec_items[0][:args.topk] for rec_label in rec_labels)
            rec_list = [rec_label in rec_items[0][:args.topk] for rec_label in rec_labels]
            
            # TH: 정답 맞춘게 있으면 바로 패스 (불필요한 연산 줄이기 위해)
            if rec_success:
                break
        
        # TH: 추천에 실패했다면, 그냥 가장 처음의 응답으로
        # if not rec_success:
        #     recommender_text = recommender_texts[0]
        rec_items_str = "".join(f"{j+1}: {id2entity[rec]}" for j, rec in enumerate(rec_items[0][:10]))

        if t == args.turn_num - 1 and last_turn_recommed:
            recommender_text = f"With that in mind, here are some recommendations: {rec_items_str}"

        if is_train:
            for target_item in target_items:
                target_name = year_pattern.sub('', target_item).strip()
                if target_name in rec_items_str:
                    rec_success = True

        seeker_prompt += f"Recommender: {recommender_text}\nSeeker:"
        seeker_full_response = chatgpt.annotate_completion(seeker_prompt).strip()
        crs_intent = seeker_full_response.split('2. Response:')[0].strip()
        seeker_text = seeker_full_response.split('2. Response:')[-1].split('Response:')[-1].strip()
        rec_success = rec_success and 'inquiry' not in crs_intent.lower()

        if not is_train:
            # Prevent from data leakage
            goal_item_list = [id2entity[idx].strip() for idx in rec_labels]
            goal_item_no_year_list = [year_pattern.sub('', id2entity[idx]).strip() for idx in rec_labels]
            seeker_response_no_movie_list = []

            for sent in nltk.sent_tokenize(seeker_text):
                use_sent = True
                for rec_item_str in goal_item_list + goal_item_no_year_list:
                    if fuzz.partial_ratio(rec_item_str.lower(), sent.lower()) > 90:
                        use_sent = False
                        break
                if use_sent is True:
                    seeker_response_no_movie_list.append(sent)

            seeker_text = ' '.join(seeker_response_no_movie_list)

        conv_dict += [{"role": "assistant", "content": recommender_text},
                      {"role": "user", "content": seeker_text}]
        seeker_prompt += f"{seeker_text}\n"

        if rec_success:
            break
    if args.evaluate:
        return conv_dict, rec_success, rec_list, original_conv_len
    else:
        return conv_dict, rec_success, original_conv_len
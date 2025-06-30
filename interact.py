
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
from few_shot import few_shot_blocks
from few_shot_new import few_shot_blocks_new, few_shot_blocks_query


instruction_prev = """You are a recommender engaging in a conversation with the user to provide recommendations.
You must follow the instructions below during the chat:

1. If you have sufficient confidence in the user's preferences, you should recommend 10 items the user is most likely to prefer without any explanations. The recommendation list can contain items that were already mentioned in the dialog. The format of the recommendation list is: no. title (year).
2. If you do not have sufficient confidence in the user's preferences, you should ask the user about their preferences.

You must either recommend or ask about the user's preferences; you must not do both simultaneously."""

instruction = """You are a recommender engaging in a conversation with the user to provide recommendations.
You must follow the instructions below during the chat:

1. If you have sufficient confidence in the user's preferences, you should recommend 10 items the user is most likely to prefer. You should analyze the user's preferences as a brief rationale. The recommendation list can contain items that were already mentioned in the dialog. The format of the recommendation list is: no. title (year). Each item should be listed with spaces.
For example: "Based on your interest in action-comedy movies with young heroes and tech geniuses, here are some recommendations: 1. Spider-Man: Homecoming (2017) 2. Guardians of the Galaxy (2014) 3. Kick-Ass (2010) 4. Scott Pilgrim vs. the World (2010) 5. The Incredibles (2004) 6. Big Hero 6 (2014) 7. Scott Pilgrim vs. the World (2010) 8. Kick-Ass 2 (2013) 9. The Incredibles 2 (2018) 10. Spider-Man: Far From Home (2019)"

2. If you do not have sufficient confidence in the user's preferences, you should ask the user about their preferences.
For example: "You like horror movies. Can you be more specific? Do you prefer classic horror, supernatural horror, slasher movies, or something else?"

Important: You must either recommend or ask about the user's preferences; you must not do both simultaneously."""

instruction_recommend = """You are a recommender engaging in a conversation with the user to provide recommendations.
You should recommend 10 items the user is most likely to prefer without any explanations. The recommendation list can contain items that were already mentioned in the dialog. The format of the recommendation list is: no. title (year)."""

instruction_query = """You are an assistant engaging in a conversation with the user to elicit her preference.
You should ask the user about her favorite genres, actors, directors, story elements, or overall impressions.
Do not recommend or mention any items while asking."""

instruction_gptcrs_prev = """You are a recommender engaging in a conversation with the user to provide recommendations.
You must follow the instructions below during the chat:

1. If you have sufficient confidence in the user's preferences, you should recommend 10 items the user is most likely to prefer without any explanations. The recommendation list can contain items that were already mentioned in the dialog. The format of the recommendation list is: no. title (year). Each item should be listed without line breaks or spaces between them.
For example, follow this format when making recommendations: "Here are some recommendations: 1. Gone with the Wind (1939)2. Pride and Prejudice (1940)3. Wuthering Heights (1939)4. Rebecca (1940)5. The Remains of the Day (1993)6. A Room with a View (1985)7. The Age of Innocence (1993)8. The English Patient (1996)9. The Remains of the Day (1993)10. Atonement (2007)"

2. If you do not have sufficient confidence in the user's preferences, you should ask the user about their preferences.
For example, you may ask: "You like horror movies. Can you be more specific? Do you prefer classic horror, supernatural horror, slasher movies, or something else?"

You must either recommend or ask about the user's preferences; you must not do both simultaneously.


"""

instruction_gptcrs = """You are a recommender engaging in a conversation with the user to provide recommendations.
You must follow the instructions below during the chat:

1. If you have sufficient confidence in the user's preferences, you should recommend 10 items the user is most likely to prefer. You should analyze the user's preferences as a brief rationale. The recommendation list can contain items that were already mentioned in the dialog. The format of the recommendation list is: no. title (year). Each item should be listed with line breaks.
Strictly follow this format when making recommendations: "Based on your interest in action-comedy movies with young heroes and tech geniuses, here are some recommendations:\n1. Spider-Man: Homecoming (2017)\n2. Guardians of the Galaxy (2014)\n3. Kick-Ass (2010)\n4. Scott Pilgrim vs. the World (2010)\n5. The Incredibles (2004)\n6. Big Hero 6 (2014)\n7. Scott Pilgrim vs. the World (2010)\n8. Kick-Ass 2 (2013)\n9. The Incredibles 2 (2018)\n10. Spider-Man: Far From Home (2019)"

2. If you do not have sufficient confidence in the user's preferences, you should ask the user about their preferences.
You may ask: "You like horror movies. Can you be more specific? Do you prefer classic horror, supernatural horror, slasher movies, or something else?"

You must either recommend or ask about the user's preferences; you must not do both simultaneously.


"""


year_pattern = re.compile(r'\(\d+\)')


def get_prompt(tokenizer, context_list, interaction_list: list = [], add_generation_prompt: bool = True, few_shot: bool = False):
    
    if not few_shot:
        return get_prompt_zeroshot(tokenizer, context_list, interaction_list, add_generation_prompt)
    else:
        return get_prompt_fewshot(tokenizer, context_list, interaction_list, add_generation_prompt)


def get_prompt_zeroshot(tokenizer, context_list, interaction_list: list = [], add_generation_prompt: bool = True):
    context = context_list[-5:]
    # context = context_list.copy()
    context.insert(0, {'role': 'system', 'content': instruction})
    context = context + interaction_list
    full_prompt = tokenizer.apply_chat_template(context, tokenize=False, add_generation_prompt=add_generation_prompt)
    
    return full_prompt


def get_prompt_fewshot(tokenizer, context_list, interaction_list: list = [], add_generation_prompt: bool = True):
    system_message = [{'role': 'system', 'content': instruction}]
    
    few_shot = []
    for example in few_shot_blocks:
        few_shot.extend(example)

    context = context_list
    full_context = system_message + few_shot + context + interaction_list

    full_prompt = tokenizer.apply_chat_template(
        full_context,
        tokenize=False,
        add_generation_prompt=add_generation_prompt
    )

    return full_prompt


def get_prompt_purpose(tokenizer, context_list, interaction_list: list = [], add_generation_prompt: bool = True, few_shot: bool = False, purpose='recommend'):
    few_shot = []

    if purpose == 'query':
        system_message = [{'role': 'system', 'content': instruction_query}]
        for example in few_shot_blocks_query:
            few_shot.extend(example)
    else:
        system_message = [{'role': 'system', 'content': instruction_recommend}]
        for example in few_shot_blocks_new:
            few_shot.extend(example)
    
    context = context_list
    full_context = system_message + few_shot + context + interaction_list

    full_prompt = tokenizer.apply_chat_template(
        full_context,
        tokenize=False,
        add_generation_prompt=add_generation_prompt
    )

    return full_prompt


def get_conv(args, model, tokenizer, full_prompt):

    # full_prompt = get_prompt(tokenizer, context, few_shot=args.few_shot)
        
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


def run_interaction(args, model, tokenizer, chatgpt, default_conv_dict, target_items, entity2id, id2entity, last_turn_recommend=False, rec_success_recommend=False, is_train=True):
    conv_dict = default_conv_dict.copy()
    original_conv_len = len(conv_dict)
    goal_item_str = ', '.join([f'"{item}"' for item in target_items])
    seeker_prompt = chatgpt.get_instruction(goal_item_str)
    for utt in conv_dict:
        seeker_prompt += f"{'Recommender' if utt['role'] == 'assistant' else 'Seeker'}: {utt['content']}\n"

    rec_success = False
    for t in range(args.turn_num):
        full_prompt = get_prompt(tokenizer, conv_dict, few_shot=args.few_shot)
        recommender_texts = get_conv(args, model, tokenizer, full_prompt)
        rec_labels = [entity2id[item] for item in target_items]

        # TH: beam이 1보다 클 경우, 응답을 하나씩 조사하여 정답을 맞춘 것을 최종 추천 응답으로 설정 (학습 시에만 해당) (최선을 다해서 추천을 해보게 함)
        # if not is_train:
        recommender_texts = [recommender_texts[0]]

        for rtxt_idx in range(len(recommender_texts)):
            recommender_text = recommender_texts[rtxt_idx]
            rec_items = chatgpt.get_rec(conv_dict, recommender_text)
            ## 수정 by BS
            rec_success = any(rec_label in rec_items[0][:args.topk] for rec_label in rec_labels)
            rec_list = [id2entity[rec_label] for rec_label in rec_labels if rec_label in rec_items[0][:args.topk] ]

            rec_ids = rec_labels
            rec_names = target_items
            topk_ids = rec_items[0][:args.topk]
            topk_names = [id2entity[item] for item in topk_ids]
            # TH: 정답 맞춘게 있으면 바로 패스 (불필요한 연산 줄이기 위해)
            if rec_success:
                break
        
        # TH: 추천에 실패했다면, 그냥 가장 처음의 응답으로
        # if not rec_success:
        #     recommender_text = recommender_texts[0]

        if (t == args.turn_num - 1 and last_turn_recommend) or (rec_success_recommend and rec_success):
            rec_items_str = "".join(f"{j+1}: {id2entity[rec]}" for j, rec in enumerate(rec_items[0][:10]))
            recommender_text = f"Here are some recommendations: {rec_items_str}"

        if is_train:
            for target_item in target_items:
                # target_name = year_pattern.sub('', target_item).strip()
                if f"accept {target_item}" in recommender_text:
                    rec_success = True

        if rec_success and args.rerank and is_train:
            if 'here are some recommendations:' in recommender_text or 'here are some more recommendations:' in recommender_text:
                if 'here are some recommendations:' in recommender_text:
                    recommendation_ment = 'here are some recommendations:'
                else:
                    recommendation_ment = 'here are some more recommendations:'

                recommendation_part = recommender_text.split(recommendation_ment)[-1]
                index_position = []
                for i in range(1, 11):
                    index_position.append(recommendation_part.find(f"{i}. "))
                index_position.append(len(recommendation_part))

                items = []
                for idx in range(len(index_position) - 1):
                    start = index_position[idx]
                    end = index_position[idx + 1]
                    item = recommendation_part[start:end]
                    items.append(item[len(f'{idx+1}. '):].strip())

                # items = recommendation_part.strip().split("\n")
                parsed_items = [item.strip() for item in items if item.strip()]

                # pattern = r'^(\d+)\.\s(.+)$'

                # parsed = [
                #     m[2] for item in parsed_items if (m := re.match(pattern, item))
                # ]

                sorted = rec_list + [i for i in parsed_items if i not in rec_list]
                final_recommendation_list = []
                for i in sorted:
                    if i not in final_recommendation_list:
                        final_recommendation_list.append(i)

                final_recommendation_list = final_recommendation_list[:10]
                if len(final_recommendation_list) < 10:
                    emb_rec_list = [id2entity[i] for i in rec_items[0] if id2entity[i] not in final_recommendation_list]
                    final_recommendation_list = final_recommendation_list + emb_rec_list
                    final_recommendation_list = final_recommendation_list[:10]

                sorted_str = ' '.join([f"{i+1}. {item}" for i, item in enumerate(final_recommendation_list[:10])])
                recommender_text = recommender_text.split(recommendation_ment)[0] + 'here are some recommendations: ' + sorted_str

        seeker_prompt += f"Recommender: {recommender_text}\nSeeker:"
        seeker_full_response = chatgpt.annotate_completion(seeker_prompt).strip()
        crs_intent = seeker_full_response.split('2. Response:')[0].strip()
        seeker_text = seeker_full_response.split('2. Response:')[-1].split('Response:')[-1].strip()
        is_recommend = 'inquiry' not in crs_intent.lower()

        if not args.hardcore:
            finish = rec_success and is_recommend
        else:
            finish = is_recommend
        
        if not is_train and args.prevent_leakage:
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

        if finish:
            break

    return conv_dict, rec_success, original_conv_len, rec_names, rec_ids, topk_names, topk_ids


def run_explore(args, model, tokenizer, chatgpt, default_conv_dict, target_items, entity2id, id2entity, last_turn_recommend=False, rec_success_recommend=False, is_train=True):
    conv_dict = default_conv_dict.copy()
    original_conv_len = len(conv_dict)
    goal_item_str = ', '.join([f'"{item}"' for item in target_items])
    seeker_prompt = chatgpt.get_instruction(goal_item_str)
    for utt in conv_dict:
        seeker_prompt += f"{'Recommender' if utt['role'] == 'assistant' else 'Seeker'}: {utt['content']}\n"

    rec_success = False
    for t in range(args.turn_num):
        full_prompt_recommend = get_prompt_purpose(tokenizer, conv_dict, few_shot=args.few_shot, purpose='recommend')
        recommender_texts = get_conv(args, model, tokenizer, full_prompt_recommend)

        rec_labels = [entity2id[item] for item in target_items]

        # TH: beam이 1보다 클 경우, 응답을 하나씩 조사하여 정답을 맞춘 것을 최종 추천 응답으로 설정 (학습 시에만 해당) (최선을 다해서 추천을 해보게 함)
        # if not is_train:
        recommender_text = recommender_texts[0]

        rec_items = chatgpt.get_rec(conv_dict, recommender_text)
        ## 수정 by BS
        rec_success = any(rec_label in rec_items[0][:args.topk] for rec_label in rec_labels)
        rec_list = [rec_label in rec_items[0][:args.topk] for rec_label in rec_labels]
        
        rec_ids = rec_labels
        rec_names = target_items
        topk_ids = rec_items[0][:args.topk]
        topk_names = [id2entity[item] for item in topk_ids]
        
        if rec_success or t == args.turn_num - 1:
            rec_items_sorted = rec_labels + [i for i in rec_items[0][:10] if i not in rec_labels]
            rec_items_str = "".join(f"{j+1}: {id2entity[rec]}" for j, rec in enumerate(rec_items_sorted[:10]))
            recommender_text = f"Here are some recommendations: {rec_items_str}"
            conv_dict += [{"role": "assistant", "content": recommender_text}]
            break
        
        full_prompt_query = get_prompt_purpose(tokenizer, conv_dict, few_shot=args.few_shot, purpose='query')
        query_texts = get_conv(args, model, tokenizer, full_prompt_query)
        query_text = query_texts[0]
        conv_dict += [{"role": "assistant", "content": query_text}]

        seeker_prompt += f"Recommender: {query_text}\nSeeker:"
        seeker_full_response = chatgpt.annotate_completion(seeker_prompt).strip()
        crs_intent = seeker_full_response.split('2. Response:')[0].strip()
        seeker_text = seeker_full_response.split('2. Response:')[-1].split('Response:')[-1].strip()

        conv_dict += [{"role": "user", "content": seeker_text}]
        seeker_prompt += f"{seeker_text}\n"

    return conv_dict, rec_success, original_conv_len, rec_names, rec_ids, topk_names, topk_ids


def run_explore_gpt(args, chatgpt, default_conv_dict, target_items, entity2id, id2entity, last_turn_recommend=False, rec_success_recommend=False, is_train=True):
    conv_dict = default_conv_dict.copy()
    original_conv_len = len(conv_dict)
    goal_item_str = ', '.join([f'"{item}"' for item in target_items])
    seeker_prompt = chatgpt.get_instruction(goal_item_str)
    crs_prompt = instruction_gptcrs
    for utt in conv_dict:
        seeker_prompt += f"{'Recommender' if utt['role'] == 'assistant' else 'Seeker'}: {utt['content']}\n"
        crs_prompt += f"{'Recommender' if utt['role'] == 'assistant' else 'Seeker'}: {utt['content']}\n"
    crs_prompt += 'Recommender: '
    rec_success = False
    rec_labels = [entity2id[item] for item in target_items]

    for t in range(args.turn_num):

        recommender_text = chatgpt.annotate_completion(crs_prompt, model_name='gpt-4.1').strip()
        rec_items = chatgpt.get_rec(conv_dict, recommender_text)
        ## 수정 by BS
        rec_success = any(rec_label in rec_items[0][:args.topk] for rec_label in rec_labels)
        rec_list = [rec_label in rec_items[0][:args.topk] for rec_label in rec_labels]
        
        rec_ids = rec_labels
        rec_names = target_items
        topk_ids = rec_items[0][:args.topk]
        topk_names = [id2entity[item] for item in topk_ids]
        
        # if rec_success or t == args.turn_num - 1:
        #     rec_items_sorted = rec_labels + [i for i in rec_items[0][:10] if i not in rec_labels]
        #     rec_items_str = "".join(f"{j+1}: {id2entity[rec]}" for j, rec in enumerate(rec_items_sorted[:10]))
        #     recommender_text = f"Here are some recommendations: {rec_items_str}"
        conv_dict += [{"role": "assistant", "content": recommender_text}]

        seeker_prompt += f"Recommender: {recommender_text}\nSeeker: "
        crs_prompt += f"{recommender_text}\nSeeker: "

        seeker_full_response = chatgpt.annotate_completion(seeker_prompt).strip()
        crs_intent = seeker_full_response.split('2. Response:')[0].strip()
        seeker_text = seeker_full_response.split('2. Response:')[-1].split('Response:')[-1].strip()
        is_recommend = 'inquiry' not in crs_intent.lower()

        conv_dict += [{"role": "user", "content": seeker_text}]
        seeker_prompt += f"{seeker_text}\n"
        crs_prompt += f"{seeker_text}\nRecommender: "

        if is_recommend and rec_success:
            break

    return conv_dict, rec_success, original_conv_len, rec_names, rec_ids, topk_names, topk_ids
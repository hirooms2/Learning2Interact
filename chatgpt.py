
from transformers import GenerationConfig
import torch
import os
import json
import typing
import openai
import numpy as np
from loguru import logger
from sklearn.metrics.pairwise import cosine_similarity
from accelerate.utils import set_seed
from tenacity import _utils, Retrying, retry_if_not_exception_type
from tenacity.stop import stop_base
from tenacity.wait import wait_base
from thefuzz import fuzz
from tqdm import tqdm


def my_before_sleep(retry_state):
    logger.debug(f'Retrying: attempt {retry_state.attempt_number} ended with: {retry_state.outcome}, spend {retry_state.seconds_since_start} in total')


## Ours prompt
seeker_instruction_template = '''You are a seeker chatting with a conversational recommender system. You prefer the following target items but are not yet aware of them. If the system recommends these target items, you must accept them.
Your target items: {}

During the chat, you must follow these instructions:

1. If the recommender suggessts {}, you must accept them. 
2. If the recommender suggessts any other items, you must refuse them without providing any explanation. You can refuse with a brief sentence such as "none of these are what I’m looking for," "I'm not interested in any of these," or other similar expressions.
3. If the recommender explicitly asks about your preferences, you should provide information related to {} without revealing the title of any target item. You must never directly mention or reveal the title of the target item. 
4. If the recommender simultaneously suggessts items and asks about your preferences, you must only accept or refuse the recommended items. You must not provide any information about your preferences.

(Tip: If the recommender says something like "Have you seen OOO?", it should be considered an item recommendation and you must only accept or refuse the recommended items)

Before responding, you must first determine the recommender's intent, which will be either recommendation or preference inquiry. Once you identify the intent, respond accordingly.

Your output must follow this format:
1. Recommender's intent: {{intent}}
2. Response: {{response}}

Your response {{response}} should not exceed approximately 30 tokens.

Here is the dialog:


'''

class my_wait_exponential(wait_base):
    def __init__(
        self,
        multiplier: typing.Union[int, float] = 1,
        max: _utils.time_unit_type = _utils.MAX_WAIT,  # noqa
        exp_base: typing.Union[int, float] = 2,
        min: _utils.time_unit_type = 0,  # noqa
    ) -> None:
        self.multiplier = multiplier
        self.min = _utils.to_seconds(min)
        self.max = _utils.to_seconds(max)
        self.exp_base = exp_base

    def __call__(self, retry_state: "RetryCallState") -> float:
        if retry_state.outcome == openai.error.Timeout:
            return 0

        try:
            exp = self.exp_base ** (retry_state.attempt_number - 1)
            result = self.multiplier * exp
        except OverflowError:
            return self.max
        return max(max(0, self.min), min(result, self.max))
    

class my_stop_after_attempt(stop_base):
    """Stop when the previous attempt >= max_attempt."""

    def __init__(self, max_attempt_number: int) -> None:
        self.max_attempt_number = max_attempt_number

    def __call__(self, retry_state: "RetryCallState") -> bool:
        if retry_state.outcome == openai.error.Timeout:
            retry_state.attempt_number -= 1
        return retry_state.attempt_number >= self.max_attempt_number
    

class ChatGPT():
    def __init__(self, args):
        self.args = args

        self.entity2id = json.load(open(os.path.join(args.home, f'data/{args.kg_dataset}/entity2id.json'),'r',encoding='utf-8'))
        self.id2info = json.load(open(os.path.join(args.home, f'data/{args.kg_dataset}/id2info.json'),'r',encoding='utf-8'))
        self.id2entityid = {}
        for id, info in self.id2info.items():
            if info['name'] in self.entity2id:
                self.id2entityid[id] = self.entity2id[info['name']]

        item_embedding_path = f"./save/embed/item/{self.args.kg_dataset}"

        item_emb_list = []
        id2item_id = []
        for i, file in tqdm(enumerate(os.listdir(item_embedding_path))):
            item_id = os.path.splitext(file)[0]
            if item_id in self.id2entityid:
                id2item_id.append(item_id)

                with open(f'{item_embedding_path}/{file}', encoding='utf-8') as f:
                    embed = json.load(f)
                    item_emb_list.append(embed)
        self.id2item_id_arr = np.asarray(id2item_id)
        self.item_emb_arr = np.asarray(item_emb_list)
        

    def annotate(self, conv_str):
        request_timeout = 6
        for attempt in Retrying(
            reraise=True, retry=retry_if_not_exception_type((openai.error.InvalidRequestError, openai.error.AuthenticationError)),
            wait=my_wait_exponential(min=1, max=60), stop=(my_stop_after_attempt(8)), before_sleep=my_before_sleep
        ):
            with attempt:
                response = openai.Embedding.create(
                    model='text-embedding-ada-002', input=conv_str, request_timeout=request_timeout
                )
            request_timeout = min(30, request_timeout * 2)

        return response
    

    def annotate_completion(self, prompt, logit_bias=None):
        if logit_bias is None:
            logit_bias = {}

        request_timeout = 20
        for attempt in Retrying(
                reraise=True,
                retry=retry_if_not_exception_type((openai.error.InvalidRequestError, openai.error.AuthenticationError)),
                wait=my_wait_exponential(min=1, max=60), stop=(my_stop_after_attempt(8))
        ):
            with attempt:
                response = openai.ChatCompletion.create(
                    model='gpt-4.1-mini', 
                    # model='gpt-4o', 
                    messages=[
                        {'role': 'user', 'content': prompt}
                    ], 
                    temperature=0, logit_bias=logit_bias,
                    request_timeout=request_timeout,
                )['choices'][0]['message']['content']
            request_timeout = min(300, request_timeout * 2)
        return response


    def get_rec(self, context_list, response=None):
        context_list = context_list + [{"role": "assisstant", "content": response}]
        conv_str = ''
        for context in context_list[-1:]:
            # conv_str += f"{context['role']}: {context['content']} "
            conv_str += f"{context['role']}: {context['content']}"
            
        conv_embed = self.annotate(conv_str)['data'][0]['embedding']
        conv_embed = np.asarray(conv_embed).reshape(1, -1)
        
        ## Cosine sim
        sim_mat = cosine_similarity(conv_embed, self.item_emb_arr)
        rank_arr = np.argsort(sim_mat, axis=-1).tolist()
        rank_arr = np.flip(rank_arr, axis=-1)[:, :50]
        item_rank_arr = self.id2item_id_arr[rank_arr].tolist()
        item_rank_arr = [[self.id2entityid[item_id] for item_id in item_rank_arr[0]]]
        
        return item_rank_arr
    

    def get_instruction(self, goal_item_str):

        seeker_instruction = seeker_instruction_template.format(goal_item_str, goal_item_str, goal_item_str)

        return seeker_instruction
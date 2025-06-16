import json
import os

from peft import LoraConfig, get_peft_model, TaskType
from parser import parse_args
from datetime import datetime
from pytz import timezone

metric = {'cnt': 0, 'hit1': 0, 'hit5': 0, 'hit10': 0}

def compute_hit(preds, label):
    for j, k in enumerate([1, 5, 10]):
        if label in preds[:k]:
            metric[f'hit{k}'] += 1

def main(args):
    log_path = os.path.join(args.home, 'results', args.log_mode, "0604161533_pretrained_beam5_fewshot_merge.json")  # 0604161533_pretrained_beam5_fewshot_merge.json
    log_path2 = os.path.join(args.home, 'results', args.log_mode, f"0614173453_sft_model_0614134510_sft_onlyinteraction_turn3_dialog-5_epoch5_batch4_gas4_lr3e-5_no_max_length_gpt_merged.json")  # 0614173453_sft_model_0614134510_sft_onlyinteraction_turn3_dialog-5_epoch5_batch4_gas4_lr3e-5_no_max_length_gpt_merged.json
    log_path3 = os.path.join(args.home, 'results', args.log_mode, f"0615115712_gpt_explore_testset_merged.json")  # 0614173453_sft_model_0614134510_sft_onlyinteraction_turn3_dialog-5_epoch5_batch4_gas4_lr3e-5_no_max_length_gpt_merged.json

    # outputs = json.load(open(log_path, 'r', encoding='utf-8'))
    given_dialogs = []
    outputs = []
    with open(log_path, encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            line = json.loads(line)
            outputs.append(line)

    outputs2 = []
    with open(log_path2, encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            line = json.loads(line)
            outputs2.append(line)

    outputs3 = []
    with open(log_path3, encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            line = json.loads(line)
            outputs3.append(line)
            given_dialogs.append(line['given_dialog'])


    sorted_idx = []
    for idx, op in enumerate(outputs):
        sorted_idx.append(given_dialogs.index(op['given_dialog']))

    outputs3 = [outputs3[i] for i in sorted_idx]

    # [(i,j) for i,j in zip(outputs, outputs2) if i['topk_ids'][0] in i['rec_ids'] and j['topk_ids'][0] not in i['rec_ids']]
    print()
    # CoT 때문에?
    # 그럼 ChatGPT도 못해야 하는거 아닌가?

    for sample in outputs:
        labels = sample['rec_ids']
        preds = sample['topk_ids']
        for label in labels:
            metric['cnt'] += 1
            compute_hit(preds, label)

    hit1 = metric['hit1'] / metric['cnt']
    hit5 = metric['hit5'] / metric['cnt']
    hit10 = metric['hit10'] / metric['cnt']
    
    print(f"cnt: {metric['cnt']}")
    print(f"hit@1 | hit@5 | hit@10")
    print(' | '.join(['%.4f' % i for i in [hit1, hit5, hit10]]))


if __name__ == "__main__":
    args = parse_args()
    main(args)

import json
import os

from peft import LoraConfig, get_peft_model, TaskType
from parser import parse_args
from datetime import datetime
from pytz import timezone

metric = {'cnt': 0, 'hit1': 0, 'hit5': 0, 'hit10': 0}
metric_by_turn = {'1': 0, '2': 0, '3': 0, '4': 0, '5': 0}

def compute_hit(preds, label):
    for j, k in enumerate([1, 5, 10]):
        if label in preds[:k]:
            metric[f'hit{k}'] += 1

def main(args):
    log_path = os.path.join(args.home, 'results', args.log_mode, f"{args.log_name}.json")
    # outputs = json.load(open(log_path, 'r', encoding='utf-8'))
    outputs = []
    with open(log_path, encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            line = json.loads(line)
            outputs.append(line)

    for sample in outputs:
        labels = sample['rec_ids']
        preds = sample['topk_ids']
        
        for label in labels:
            metric['cnt'] += 1
            compute_hit(preds, label)
            
            
            if label not in preds:
                continue
            gen_length = len(sample['generated_dialog'])//2
            if str(gen_length) in metric_by_turn:
                metric_by_turn[str(gen_length)] += 1
            
    total_hit_cnt = 0
    for i in range(1, 6):
        total_hit_cnt += metric_by_turn[str(i)]


    hit1 = metric['hit1'] / metric['cnt']
    hit5 = metric['hit5'] / metric['cnt']
    hit10 = metric['hit10'] / metric['cnt']
    
    print(f"cnt: {metric['cnt']}")
    print(f"hit@1 | hit@5 | hit@10")
    print(' | '.join(['%.4f' % i for i in [hit1, hit5, hit10]]))
    print(f"\nTotal: {total_hit_cnt}")
    print("Turn@1 | Turn@2 | Turn@3 | Turn@4 | Turn@5")
    print('  |  '.join([str(metric_by_turn[str(i)]) for i in range(1, 6)]))


if __name__ == "__main__":
    args = parse_args()
    main(args)

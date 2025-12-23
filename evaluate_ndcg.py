import json
import os
import math
from parser import parse_args

def compute_ndcg(preds, label):
    if label in preds[:10]:
        rank = preds.index(label) + 1  
        dcg = 1.0 / math.log2(rank + 1)
        idcg = 1.0  
        return dcg / idcg
    else:
        return 0.0

def main(args):
    log_path = os.path.join(args.home, 'results', args.log_mode, f"{args.log_name}.json")
    outputs = []
    with open(log_path, encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            line = json.loads(line)
            outputs.append(line)

    cnt = 0
    total_ndcg = 0.0

    for sample in outputs:
        labels = sample['rec_ids']
        preds = sample['topk_ids']

        for label in labels:
            cnt += 1
            total_ndcg += compute_ndcg(preds, label)

    avg_ndcg = total_ndcg / cnt if cnt > 0 else 0.0

    print(f"cnt: {cnt}")
    print(f"nDCG@10: {avg_ndcg:.4f}")

if __name__ == "__main__":
    args = parse_args()
    main(args)

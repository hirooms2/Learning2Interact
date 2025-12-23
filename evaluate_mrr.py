import json
import os

from parser import parse_args

metric = {'cnt': 0, 'mrr': 0.0}
metric_by_turn = {'1': 0.0, '2': 0.0, '3': 0.0, '4': 0.0, '5': 0.0}

def compute_mrr(preds, label):
    if label in preds:
        rank = preds.index(label) + 1   # index는 0부터 시작하므로 +1
        return 1.0 / rank
    return 0.0

def main(args):
    log_path = os.path.join(args.home, 'results', args.log_mode, f"{args.log_name}.json")
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
            rr = compute_mrr(preds, label)
            metric['mrr'] += rr

            if rr > 0:  # 정답이 예측 안에 들어있을 때만 turn별 MRR 집계
                gen_length = len(sample['generated_dialog']) // 2
                if str(gen_length) in metric_by_turn:
                    metric_by_turn[str(gen_length)] += rr

    # 평균 계산
    mrr = metric['mrr'] / metric['cnt'] if metric['cnt'] > 0 else 0.0
    print(f"cnt: {metric['cnt']}")
    print(f"MRR: {mrr:.4f}")

    print("\nTurn@1 | Turn@2 | Turn@3 | Turn@4 | Turn@5")
    print('  |  '.join([f"{metric_by_turn[str(i)]:.4f}" for i in range(1, 6)]))


if __name__ == "__main__":
    args = parse_args()
    main(args)

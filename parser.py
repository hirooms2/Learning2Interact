import argparse
import os


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_new_tokens', type=int, default=256)
    parser.add_argument('--num_beams', type=int, default=1)
    parser.add_argument('--kg_dataset', type=str, default='redial')
    parser.add_argument('--turn_num', type=int, default=5)
    parser.add_argument('--topk', type=int, default=10)
    parser.add_argument('--api_key', type=str, default='')
    parser.add_argument('--model_path', type=str, default='')

    args = parser.parse_args()

    from platform import system as sysChecker
    if sysChecker() == 'Linux':
        args.home = os.path.dirname(__file__)
    elif sysChecker() == "Windows":
        args.home = ''
    print(args.home)

    if args.model_path != '':
        args.model_path = os.path.join(args.home, 'model_weights', args.model_path)
        
    return args
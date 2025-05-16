import argparse
import os


def parse_args():
    parser = argparse.ArgumentParser()

    # Log
    parser.add_argument('--log_name', type=str, default='log_name')

    # Generation
    parser.add_argument('--max_new_tokens', type=int, default=256)
    parser.add_argument('--num_beams', type=int, default=1)
    # Dataset
    parser.add_argument('--kg_dataset', type=str, default='redial')
    parser.add_argument('--train_data', type=str, default='redial_processed_train.json')
    parser.add_argument('--test_data', type=str, default='redial_processed_test.json')
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--end', type=int, default=5000)
    # Interaction Setting
    parser.add_argument('--turn_num', type=int, default=5)
    parser.add_argument('--topk', type=int, default=10)
    parser.add_argument('--last_turn_recommed', action='store_true')

    # Parameter
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--eval_batch_size', type=int, default=4)
    parser.add_argument('--epoch', type=int, default=2)
    parser.add_argument('--ppo_epoch', type=int, default=4)
    parser.add_argument('--learning_rate', type=float, default=3e-5)
    parser.add_argument('--logging_steps', type=int, default=100)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1)
    
    # PPO tuning
    parser.add_argument('--init_kl_coef', type=float, default=0.2)
    parser.add_argument('--gamma', type=float, default=0.9)
    parser.add_argument('--reward', type=float, default=1.0)
    parser.add_argument('--adap_kl_ctrl', action='store_true')
    
    # ChatGPT API
    parser.add_argument('--api_key', type=str, default='')
    # CRS Model
    parser.add_argument('--model_name', type=str, default='meta-llama/Meta-Llama-3.1-8B-Instruct')
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
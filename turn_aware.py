from parser import parse_args
import re
import os

def count_assistant_user_pairs(block):
    parts = re.split(r'-{10,}', block)
    if len(parts) < 2:
        return 0
    main_dialog = parts[1]
    assistant_lines = re.findall(r'^assistant:', main_dialog, re.MULTILINE)
    user_lines = re.findall(r'^user:', main_dialog, re.MULTILINE)
    return min(len(assistant_lines), len(user_lines))

def split_success_cases_by_turn(input_path, output_path_turn3, output_path_turn45):
    with open(input_path, 'r', encoding='utf-8') as f:
        content = f.read()

    dialog_blocks = re.findall(r'(#{33,} Dialog Case \d+ #{33,}.*?)#{3,}', content, re.DOTALL)

    case_turn3 = []
    case_turn45 = []

    for block in dialog_blocks:
        if '[[[REC_SUCCESS: True]]]' in block:
            pair_count = count_assistant_user_pairs(block)
            if pair_count <= 3:
                case_turn3.append(block.strip())
            elif 4 <= pair_count <= 5:
                case_turn45.append(block.strip())

    with open(output_path_turn3, 'w', encoding='utf-8') as f:
        f.write('\n\n'.join(case_turn3))

    with open(output_path_turn45, 'w', encoding='utf-8') as f:
        f.write('\n\n'.join(case_turn45))

    print(f"[✓] Saved {len(case_turn3)} cases with ≤3 turns to {output_path_turn3}")
    print(f"[✓] Saved {len(case_turn45)} cases with 4~5 turns to {output_path_turn45}")

def main():
    args = parse_args()

    log_path = os.path.join(args.home, 'results', args.log_mode, f"{args.log_name}.txt")

    if not os.path.exists(log_path):
        print(f"[X] Log file '{log_path}' does not exist.")
        return

    output_turn3 = f"{args.log_name}_turn_le3.txt"
    output_turn45 = f"{args.log_name}_turn_4to5.txt"

    split_success_cases_by_turn(log_path, output_turn3, output_turn45)

if __name__ == '__main__':
    main()

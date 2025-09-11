import json

def convert_json(input_path, output_path):
    results = []

    # JSONL 형식 처리
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            sample = json.loads(line)

            # rec_success == True 인 경우만 변환
            #if not sample.get("rec_success", False):
            #    continue

            new_entry = {
                "dialog": sample.get("given_dialog", []),
                "interaction": sample.get("generated_dialog", []),
                "target_items": sample.get("rec_names", [])
            }
            results.append(new_entry)

    # 변환된 데이터 저장 (JSON 배열 형식)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)

# 실행 예시
if __name__ == "__main__":
    convert_json(
        r"/home/user/junpyo/Learning2Interact/make_sft/0906152045_opendialkg_explore_gpt41_trainset_merge.json",
        r"/home/user/junpyo/Learning2Interact/data/opendialkg_processed_train_sft_gpt_all.json"
    )

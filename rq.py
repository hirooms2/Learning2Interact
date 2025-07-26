import json

# Merge 한 json 파일 읽어서 데이터 전처리 => 전체, 성공, 실패 데이터로 나누기
def make_data(data_path):
    with open(data_path, 'r', encoding='utf-8') as f:
        data = []
        for line in f:
            data.append(json.loads(line))
        data_suc = []
        data_fal = []
        for item in data:
            if item['rec_success']:
                data_suc.append(item)
            else:
                data_fal.append(item)
    return data, data_suc, data_fal   
 
# 전체 발화중 추천발화 비율 계산
def rec_ratio(data):
    rec_counter = 0
    utt_counter = 0
    for item in data :
        utt_counter += (len(item['generated_dialog']))/2
        for utt in item['generated_dialog'] :
            if utt['role'] == 'assistant' :
                if "recommendations:" in utt['content']:
                    rec_counter +=1
    print(f"rec_ratio = {rec_counter/utt_counter}")

# 성공 케이스의 평균 턴 수 계산
def avg_turn(data):
    suc_counter = 0
    utt_counter = 0
    for item in data:
        if item['rec_success']:
            suc_counter +=1
            utt_counter += (len(item['generated_dialog']))/2
    
    print(f"avg_turn = {utt_counter/suc_counter}")




if __name__ == "__main__":
    data_path = r"C:\Users\user\Desktop\새 폴더\data\0719030855_grpo_0717220457_turn2_last2_hardcore_S1000_merge.json"
    data, data_suc, data_fal = make_data(data_path)
    rec_ratio(data)
    avg_turn(data)
    
    data_path = r"C:\Users\user\Desktop\새 폴더\data\0716103434_grpo_turn510epoch_S1000_merge.json"
    data, data_suc, data_fal = make_data(data_path)
    rec_ratio(data)
    avg_turn(data)
import json


prev_data = json.load(open('/home/user/junpyo/Learning2Interact/make_sft/0616155421_gpt_explore2_trainset_merge.json', 'r', encoding='utf-8'))
given_dialogs = [i['given_dialog'] for i in prev_data]

current_data = json.load(open('/home/user/junpyo/Learning2Interact/make_sft/0619233833_sft_rerank3_batch32_g1_coef00_lam1_naive_E1.json', 'r', encoding='utf-8'))

win = []
draw = []
lose = []

for data in current_data:
    idx = given_dialogs.index(data['given_dialog'])
    if prev_data[idx]['rec_success'] is False and data['REC_SUCCESS'] is True:
        win.append(data)
    elif prev_data[idx]['rec_success'] is True and data['REC_SUCCESS'] is False:
        lose.append(data)
    else:
        draw.append(data)
    
print()
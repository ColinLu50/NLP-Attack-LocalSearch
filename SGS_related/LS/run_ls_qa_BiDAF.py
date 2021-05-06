import sys

sys.path.append(sys.path[0] + '/../../')

import json
import argparse


from SGS_related.LS.ls_qa import LSBidafQA


from SGS_related.log_manager import QALogManager
from SGS_related.shared_params import RESULT_FOLDER

from utils import my_file


dataset_path = '../processed_dataset/dev-v2.0.json'

parser = argparse.ArgumentParser(description='Morpheus.')
parser.add_argument("--model", "-m", type=str, required=True, choices=['bidaf', 'elmobidaf'])

args = parser.parse_args()

data = json.load(open(dataset_path))

# total_model_queries = 0

model_type = args.model  # 'bidaf', 'elmobidaf'
alg_ = 'ls'
dataset_ = 'sqaud2'

attacker = LSBidafQA(model_type)

out_folder_name = f'{model_type}_{alg_}'
# all_loggers = QALogManager(out_folder_name, 'all_logs.txt')
answerable_loggers = QALogManager(out_folder_name, 'answerable_logs.txt')

all_q_num = 0
skip_num = 0

for i, article in enumerate(data['data']):
    para = 0
    num_paras = len(article['paragraphs'])
    for j, paragraph in enumerate(article['paragraphs']):
        context = paragraph['context']
        for k, qa in enumerate(paragraph['qas']):
            # only attack answerable
            is_answerable = not qa['is_impossible']

            if not is_answerable:
                continue

            all_q_num += 1
            attack_result, num_queries, modif_rate, is_attack_success = attacker.attack_one(qa,
                                                                                       context,
                                                                                       constrain_pos=True)

            # recorde result
            data['data'][i]['paragraphs'][j]['qas'][k]['question'] = attack_result

            # check if original question satisfies the requirements: 1. correct 2. length 10-100
            if modif_rate is None:
                skip_num += 1
                continue

            # log result
            # all_loggers.log_attack_result(is_attack_success, modif_rate, num_queries)
            # if is_answerable:
            answerable_loggers.log_attack_result(is_attack_success, modif_rate, num_queries)
    # break

print(f'\n all number: {all_q_num}, skip number: {skip_num}')

# summary
# print('================= all attack summary =====================')
# all_loggers.summary()

print('================= answerable attack summary =====================')
answerable_loggers.summary()

with open(my_file.real_path_of(RESULT_FOLDER, out_folder_name, f'{dataset_}_{model_type}_{alg_}.json'), 'w') as f:
    json.dump(data, f, indent=4)

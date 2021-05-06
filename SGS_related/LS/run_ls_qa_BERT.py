import sys

sys.path.append(sys.path[0] + '/../../')

import json
import argparse


from SGS_related.LS.ls_qa import LSHuggingfaceBertQA

from SGS_related.log_manager import QALogManager
from SGS_related.shared_params import RESULT_FOLDER

from utils import my_file



dataset_path = '../processed_dataset/dev-v2.0.json'


parser = argparse.ArgumentParser(description='Morpheus.')
parser.add_argument("--model", "-m", type=str, required=True, choices=['bert', 'spanbert'])
args = parser.parse_args()


data = json.load(open(dataset_path))

# total_model_queries = 0

model_type = args.model #
if model_type == 'bert':
    print('using BERT')
    model_path = "twmkn9/albert-base-v2-squad2"
    tokenizer_path = "twmkn9/albert-base-v2-squad2"
    model_ = 'BERT'
elif model_type == 'spanbert':
    print('using SpanBERT')
    model_path = 'mrm8488/spanbert-large-finetuned-squadv2'
    tokenizer_path = "SpanBERT/spanbert-large-cased"
    model_ = 'SpanBERT'

attacker = LSHuggingfaceBertQA(model_path, tokenizer_path, squad2=True)

alg_ = 'ls'
dataset_ = 'sqaud2'

out_folder_name = f'{model_}_{alg_}'
all_loggers = QALogManager(out_folder_name, 'all_logs.txt')
answerable_loggers = QALogManager(out_folder_name, 'answerable_logs.txt')

all_q_num = 0
skip_num = 0

for i, article in enumerate(data['data']):
    para = 0
    num_paras = len(article['paragraphs'])
    for j, paragraph in enumerate(article['paragraphs']):
        context = paragraph['context']
        for k, qa in enumerate(paragraph['qas']):
            all_q_num += 1

            is_answerable = not qa['is_impossible']

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
            all_loggers.log_attack_result(is_attack_success, modif_rate, num_queries)
            if is_answerable:
                answerable_loggers.log_attack_result(is_attack_success, modif_rate, num_queries)


print(f'\n all number: {all_q_num}, skip number: {skip_num}')

# summary
print('================= all attack summary =====================')
all_loggers.summary()

print('================= answerable attack summary =====================')
answerable_loggers.summary()


with open(my_file.real_path_of(RESULT_FOLDER, out_folder_name, f'{dataset_}_{model_}_{alg_}.json'), 'w') as f:
    json.dump(data, f, indent=4)

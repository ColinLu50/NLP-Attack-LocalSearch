import sys

sys.path.append(sys.path[0] + '/../../')
import os

import argparse
from SGS_related.LS.ls_nmt import LSFairseqTransformerNMT

from SGS_related.log_manager import QALogManager
from SGS_related.shared_params import RESULT_FOLDER, ConvS2S_PATH, Transformer_PATH

from utils import my_file

parser = argparse.ArgumentParser()
parser.add_argument("--model", "-m", type=str, required=True, choices=['convS2S', 'transformer'])

args = parser.parse_args()


model_type = args.model
if model_type == 'convS2S':
    model_dir = ConvS2S_PATH
    model_file = 'model.pt'
elif model_type == 'transformer':
    model_dir = Transformer_PATH
    model_file = 'model.pt'

model_ = model_type
alg_ = 'ls'
dataset_ = 'newstest'

out_folder_name = f'{model_}_{alg_}'


newstest_file_path = '../processed_dataset/'
src_filename = 'newstest2014-en.txt'
tgt_filename = 'newstest2014-fr.txt'

src_file = os.path.join(newstest_file_path, src_filename)
tgt_file = os.path.join(newstest_file_path, tgt_filename)

# print('Writing to ' + out_file)

attacker = LSFairseqTransformerNMT(model_dir, model_file)

all_loggers = QALogManager(out_folder_name, 'logs.txt')


all_num = 0
skip_num = 0



_cnt = 0

res_pairs = []

with open(src_file, 'r') as src_stream, open(tgt_file, 'r') as tgt_stream:

    for src, ref in zip(src_stream, tgt_stream):

        all_num += 1

        adv, adv_pred, is_attack_success, query_number, modif_rate = attacker.attack_one(src, ref)

        # check if original question satisfies the requirements: 1. correct 2. length 10-100
        if is_attack_success is None:
            skip_num += 1
            continue


        all_loggers.log_attack_result(is_attack_success, modif_rate, query_number)

        res_pairs.append((is_attack_success, src, ref, adv, adv_pred, modif_rate))


print(f'{skip_num}/{all_num}')
all_loggers.summary()

my_file.save_pkl_in_repo(res_pairs, RESULT_FOLDER, out_folder_name, f'results_{model_}_{alg_}.pkl')
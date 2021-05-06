import os
import sys

sys.path.append(sys.path[0] + '/../../')


from time import time
from datetime import datetime
import numpy as np

from PSO_related.snli.attack_pso_snli import PSOAttack

from PSO_related.snli.encap_snli_bert import Model
from PSO_related.all_shared.logger import MyLogger
from PSO_related.all_shared.outputs_saver import ResultSaver
from utils import my_file

# ============= param ========================
TEST_SIZE = None # None: attack all examples. Postive Number: attack TEST_SIZE examples
SEED = 3333
pop_size = 60
max_iter = 20

SNLI_data_folder_path = '/home/workspace/nlp_attack/data/pso_raw/SNLI_used_data'
# ===========================================


np.random.seed(SEED)

dataset_name = 'SNLI'
model_name = 'BERT'
tag = 'PSO'
SAVE_FOLDER = f'out/pso_related/{dataset_name}_{model_name}_{tag}_search/{SEED}'
my_file.create_folder(SAVE_FOLDER)

# init log file
log_file = open(my_file.real_path_of(SAVE_FOLDER, 'log.txt'), 'w')

# save parametes
log_file.write(f'SEED: {SEED}\n')
log_file.write(f'Test Size: {TEST_SIZE}\n')
log_file.write(f'Pop size: {pop_size}\n')
log_file.write(f'Max Iteration: {max_iter}\n')
log_file.flush()

tokenizer = my_file.load_pkl(os.path.join(SNLI_data_folder_path, 'nli_tokenizer.pkl'))
word_candidate = my_file.load_pkl(os.path.join(SNLI_data_folder_path, 'word_candidates_sense.pkl'))
train, valid, test = my_file.load_pkl(os.path.join(SNLI_data_folder_path, 'all_seqs.pkl'))
test_pos_tags = my_file.load_pkl(os.path.join(SNLI_data_folder_path, 'pos_tags_test.pkl'))

test_s1 = [t[1:-1] for t in test['s1']]
test_s2 = [t[1:-1] for t in test['s2']]

vocab = {w: i for (w, i) in tokenizer.word_index.items()}
inv_vocab = {i: w for (w, i) in vocab.items()}
model = Model(inv_vocab, os.path.join(SNLI_data_folder_path, 'BERTModel.pt'))

adversary = PSOAttack(model, word_candidate, pop_size=pop_size, max_iters=max_iter)
print('the length of test cases is:', len(test_s1))

test_idxs = np.arange(len(test_s1))
# np.random.shuffle(test_idxs)
# test_idxs = np.random.choice(len(test_s1), size=len(test_s1), replace=False)

test_list = []

skip_num = 0
total_num = 0

# logger
cur_logger = MyLogger(log_file=log_file)

# saver
cur_saver = ResultSaver()

print('start')
for test_idx in test_idxs:
    total_num += 1
    s1 = test_s1[test_idx]
    s2 = test_s2[test_idx]
    pos_tags = test_pos_tags[test_idx]
    attack_pred = np.argmax(model.pred([s1], [s2])[0])
    true_label = test['label'][test_idx]
    x_len = np.sum(np.sign(s2))
    if attack_pred != true_label:
        print('Wrong classified')
        skip_num += 1
    elif x_len < 10:
        print('Skipping too short input')
    else:
        if true_label == 2:
            target = 0
        elif true_label == 0:
            target = 2
        else:
            target = 0 if np.random.uniform() < 0.5 else 2

        test_list.append(test_idx)
        # attack
        attack_result = adversary.attack(s1, s2, target, pos_tags)

        cur_logger.log_result(attack_result, s2, x_len, adversary)
        cur_saver.record_attack(attack_result, s2, x_len, test_idx, target)

        if TEST_SIZE and (len(test_list) >= TEST_SIZE):
            break

cur_logger.summary()
cur_saver.save_to_folder(SAVE_FOLDER)

log_file.close()
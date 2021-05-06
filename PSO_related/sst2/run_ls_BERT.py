from __future__ import division

import sys

sys.path.append(sys.path[0] + '/../../')

from datetime import datetime
import numpy as np
# import tensorflow as tf

from keras.preprocessing.sequence import pad_sequences

from PSO_related.imdb_sst2_shared.attack_ls import LSAttack

import PSO_related.sst2.encap_sst_bert_zy as models
from PSO_related.imdb_sst2_shared.check_result import check_attack_result
from PSO_related.all_shared.outputs_saver import ResultSaver
from PSO_related.all_shared.logger import MyLogger
from utils import my_file

# ============= param ========================
TEST_SIZE = None # None: attack all examples. Postive Number: attack TEST_SIZE examples
SEED = 3333

dataset_path = '/home/workspace/nlp_attack/data/pso_raw/SST_used_data/aux_files/dataset_13837.pkl'
word_candidates_path = '/home/workspace/nlp_attack/data/pso_raw/SST_used_data/word_candidates_sense.pkl'
pos_tags_path = '/home/workspace/nlp_attack/data/pso_raw/SST_used_data/pos_tags_test.pkl'
model_path = '/home/workspace/nlp_attack/data/pso_raw/SST_used_data/BERTModel.pt'


np.random.seed(SEED)

dataset_name = 'SST2'
model_name = 'BERT'
tag = 'LS'
SAVE_FOLDER = f'out/pso_related/{dataset_name}_{model_name}_{tag}_search/{SEED}'
my_file.create_folder(SAVE_FOLDER)

# init log file
log_file = open(my_file.real_path_of(SAVE_FOLDER, 'log.txt'), 'w')

# save parametes
log_file.write(f'SEED: {SEED}\n')
log_file.write(f'Test Size: {TEST_SIZE}\n')
log_file.flush()


dataset = my_file.load_pkl_in_repo(dataset_path)
word_candidate = my_file.load_pkl_in_repo(word_candidates_path)
test_pos_tags = my_file.load_pkl_in_repo(pos_tags_path)

# Prevent returning 0 as most similar word because it is not part of the dictionary
max_len = 250
train_x = pad_sequences(dataset.train_seqs2, maxlen=max_len, padding='post')
train_y = np.array(dataset.train_y)
test_x = pad_sequences(dataset.test_seqs2, maxlen=max_len, padding='post')
test_y = np.array(dataset.test_y)

model = models.Model(dataset, model_path).cuda()
attacker = LSAttack(model, word_candidate)

all_test_num = len(dataset.test_y)
print(f'Total have {all_test_num} test examples')

test_list = []
# orig_list = []
# orig_label_list = []
# adv_list = []
# dist_list = []
# adv_orig = []
# adv_orig_label = []
# fail_list = []
# adv_training_examples = []

wrong_classified_num = 0
total_num = 0

cur_logger = MyLogger(log_file)
cur_saver = ResultSaver()

for test_idx in range(all_test_num):
    total_num += 1
    pos_tags = test_pos_tags[test_idx]
    x_orig = test_x[test_idx]
    orig_label = test_y[test_idx]
    orig_preds = model.predict(x_orig[np.newaxis, :])[0]
    if np.argmax(orig_preds) != orig_label:
        print('skipping wrong classifed ..')
        print('--------------------------')
        wrong_classified_num += 1
        continue
    x_len = np.sum(np.sign(x_orig))
    if x_len >= 100:
        print('skipping too long input..')
        print('--------------------------')
        continue
    if x_len < 10:
        print('skipping too short input..')
        print('--------------------------')
        continue

    test_list.append(test_idx)
    # orig_list.append(x_orig)
    target_label = 1 if orig_label == 0 else 0
    # orig_label_list.append(orig_label)
    x_adv = attacker.attack(x_orig, target_label, pos_tags)

    check_attack_result(x_adv, x_orig, target_label, attacker)

    cur_logger.log_result(x_adv, x_orig, x_len, attacker)
    cur_saver.record_attack(x_adv, x_orig, x_len, test_idx, target_label)

    if TEST_SIZE and (len(test_list) >= TEST_SIZE):
        break

cur_logger.summary()
cur_saver.save_to_folder(SAVE_FOLDER)

log_file.close()


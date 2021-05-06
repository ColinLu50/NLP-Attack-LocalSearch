from __future__ import division

import sys

sys.path.append(sys.path[0] + '/../../')

from datetime import datetime
import numpy as np

from keras.preprocessing.sequence import pad_sequences

import PSO_related.imdb.encap_imdb_bert_zy as models
from PSO_related.imdb_sst2_shared.attack_pso import PSOAttack
from PSO_related.all_shared.logger import MyLogger
from PSO_related.all_shared.outputs_saver import ResultSaver
from utils import my_file
# from src.pso_reimplements.pso_shared import SUCCESS_THRESHOLD


# ============= param ========================
TEST_SIZE = None # None: attack all examples. Postive Number: attack TEST_SIZE examples
SEED = 3232
pop_size = 60
max_iter = 20

# data path
dataset_path = '/home/workspace/nlp_attack/data/pso_raw/IMDB_used_data/aux_files/dataset_50000.pkl'
word_candidates_path = '/home/workspace/nlp_attack/data/pso_raw/IMDB_used_data/word_candidates_sense.pkl'
pos_tags_path = '/home/workspace/nlp_attack/data/pso_raw/IMDB_used_data/pos_tags_test.pkl'
model_path = '/home/workspace/nlp_attack/data/pso_raw/IMDB_used_data/BERTModel.pt'


# ============================================

np.random.seed(SEED)
# tf.set_random_seed(3333)


dataset_name = 'IMDB'
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

# CURRENT_PATH = 'data/pso_raw/IMDB_used_data'
VOCAB_SIZE = 50000
dataset = my_file.load_pkl(dataset_path)
word_candidate = my_file.load_pkl(word_candidates_path)
test_pos_tags = my_file.load_pkl(pos_tags_path)

# Prevent returning 0 as most similar word because it is not part of the dictionary
max_len = 250
train_x = pad_sequences(dataset.train_seqs2, maxlen=max_len, padding='post')
train_y = np.array(dataset.train_y)
test_x = pad_sequences(dataset.test_seqs2, maxlen=max_len, padding='post')
test_y = np.array(dataset.test_y)

model = models.Model(dataset, model_path).cuda()
pso_attacker = PSOAttack(model, word_candidate, dataset,
                         max_iters=max_iter,
                         pop_size=pop_size)


all_test_num = len(dataset.test_y)
print(f'Total have {all_test_num} test examples')


skip_num = 0
total_num = 0

# logger
cur_logger = MyLogger(log_file)

# save results
test_idx_list = []

cur_saver = ResultSaver()


log_file.write('======================START======================\n\n')
for test_idx in range(all_test_num):
    total_num += 1
    pos_tags = test_pos_tags[test_idx]
    x_orig = test_x[test_idx]
    orig_label = test_y[test_idx]
    orig_preds = model.predict(x_orig[np.newaxis, :])[0]
    if np.argmax(orig_preds) != orig_label:
        print('skipping wrong classifed ..')
        print('--------------------------')
        skip_num += 1
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

    test_idx_list.append(test_idx)
    target_label = 1 if orig_label == 0 else 0
    x_adv = pso_attacker.attack(x_orig, target_label, pos_tags)

    cur_logger.log_result(x_adv, x_orig, x_len, pso_attacker)
    cur_saver.record_attack(x_adv, x_orig, x_len, test_idx, target_label)


    if TEST_SIZE and (len(test_idx_list) >= TEST_SIZE):
        break

cur_logger.summary()
cur_saver.save_to_folder(SAVE_FOLDER)

log_file.close()


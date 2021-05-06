from __future__ import division

import sys

sys.path.append(sys.path[0] + '/../../')
from datetime import datetime

import numpy as np
import tensorflow as tf

from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import Embedding, CuDNNLSTM, Bidirectional, GlobalMaxPooling1D
from keras.preprocessing.sequence import pad_sequences

from  PSO_related.imdb_sst2_shared.attack_pso import PSOAttack

from PSO_related.all_shared.logger import MyLogger
from PSO_related.all_shared.outputs_saver import ResultSaver
from utils import my_file

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)

# ============= param ========================
TEST_SIZE = None # None: attack all examples. Postive Number: attack TEST_SIZE examples
SEED = 3232

pop_size = 60
max_iter = 20

dataset_path = '/home/workspace/nlp_attack/data/pso_raw/IMDB_used_data/aux_files/dataset_50000.pkl'
word_candidates_path = '/home/workspace/nlp_attack/data/pso_raw/IMDB_used_data/word_candidates_sense.pkl'
pos_tags_path = '/home/workspace/nlp_attack/data/pso_raw/IMDB_used_data/pos_tags_test.pkl'

glove_path = '/home/workspace/nlp_attack/data/pso_raw/IMDB_used_data/aux_files/embeddings_glove_50000.npy'
model_path = '/home/workspace/nlp_attack/data/pso_raw/IMDB_used_data/bdlstm_models'

# ===========================================

np.random.seed(SEED)
tf.set_random_seed(SEED)

dataset_name = 'IMDB'
model_name = 'BiLSTM'
tag = 'PSO'
time_str = datetime.now().strftime('%y%m%d_%H%M')
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

VOCAB_SIZE = 50000
dataset = my_file.load_pkl_in_repo(dataset_path)
word_candidate = my_file.load_pkl_in_repo(word_candidates_path)
test_pos_tags = my_file.load_pkl_in_repo(pos_tags_path)

# Prevent returning 0 as most similar word because it is not part of the dictionary
max_len = 250
train_x = pad_sequences(dataset.train_seqs2, maxlen=max_len, padding='post')
train_y = np.array(dataset.train_y)
test_x = pad_sequences(dataset.test_seqs2, maxlen=max_len, padding='post')
test_y = np.array(dataset.test_y)

# max_len =  100

pop_size = 60


def bd_lstm(embedding_matrix):
    max_len = 250
    num_classes = 2
    loss = 'binary_crossentropy'
    activation = 'sigmoid'
    embedding_dims = 300
    num_words = VOCAB_SIZE
    print('Build word_bdlstm model...')
    model = Sequential()
    model.add(Embedding(  # Layer 0, Start
        input_dim=num_words + 1,  # Size to dictionary, has to be input + 1
        output_dim=embedding_dims,  # Dimensions to generate
        weights=[embedding_matrix],  # Initialize word weights
        input_length=max_len,
        name="embedding_layer",
        trainable=False))
    OPTIMIZER = 'adam'

    model.add(Bidirectional(CuDNNLSTM(128, return_sequences=True)))
    model.add(Bidirectional(CuDNNLSTM(64, return_sequences=True)))
    model.add(Dropout(0.5))
    model.add(GlobalMaxPooling1D())
    model.add(Dense(2, activation=activation))
    model.summary()

    # try using different optimizers and different optimizer configs
    model.compile(OPTIMIZER, loss, metrics=['accuracy'])
    return model


embedding_matrix = np.load(glove_path)
embedding_matrix = embedding_matrix.T
model = bd_lstm(embedding_matrix)
model.load_weights(model_path)

test_y2 = np.array([[0, 1] if t == 1 else [1, 0] for t in test_y])
all_scores_origin = model.evaluate(test_x, test_y2)
print('all origin test_loss: %f, accuracy: %f' % (all_scores_origin[0], all_scores_origin[1]))
log_file.write('all origin test_loss: %f, accuracy: %f' % (all_scores_origin[0], all_scores_origin[1]))
pso_attacker = PSOAttack(model, word_candidate, dataset,
                         max_iters=max_iter,
                         pop_size=pop_size)

all_test_num = len(dataset.test_y)
print(f'Total have {all_test_num} test examples')

# orig_list = []
# orig_label_list = []


success_num = 0

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


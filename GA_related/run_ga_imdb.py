import os
import sys
sys.path.append(sys.path[0] + '/../')

from time import time
import numpy as np
import tensorflow as tf
import pickle

from keras.preprocessing.sequence import pad_sequences

from GA_related import models
# from src.ga_reimp.imdb import display_utils

from GA_related.shared.goog_lm import LM
from GA_related.shared.attacks_GA import GeneticAtack
from GA_related.shared.ga_logger import GAIMDBLogger
from GA_related.shared.ga_recorder import GARecorderIMDB

from utils import my_file

# tf allow grouth
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)

GA_OUT_PATH = 'out/GA_related/'

# ============== Paras =========================
# 5555, 6666, 7777, 8888, 9999, 1010, 111, 222, 333
SEED = 7777
TEST_SIZE = None

pop_size = 60
max_iters = 20
n1 = 8
K = 4

algo = 'EMLM_GA'
dataset_name = 'IMDB'

np.random.seed(SEED)
tf.set_random_seed(SEED)

# load dataset
VOCAB_SIZE = 50000
with open('aux_files/dataset_%d.pkl' %VOCAB_SIZE, 'rb') as f:
    dataset = pickle.load(f)
# dataset = my_file.load_pkl_in_repo(GA_IMDB_ALL_PATH, 'aux_files/dataset_%d.pkl' % VOCAB_SIZE)

doc_len = [len(dataset.test_seqs2[i]) for i in
           range(len(dataset.test_seqs2))]

npy_path1 = 'aux_files/dist_counter_%d.npy' % VOCAB_SIZE
dist_mat = np.load(npy_path1)
# Prevent returning 0 as most similar word because it is not part of the dictionary
dist_mat[0, :] = 100000
dist_mat[:, 0] = 100000

npy_path2 = 'aux_files/missed_embeddings_counter_%d.npy' % VOCAB_SIZE
skip_list = np.load(npy_path2)


# Preparing the dataset
max_len = 250
train_x = pad_sequences(dataset.train_seqs2, maxlen=max_len, padding='post')
train_y = np.array(dataset.train_y)
test_x = pad_sequences(dataset.test_seqs2, maxlen=max_len, padding='post')
test_y = np.array(dataset.test_y)

# Loading the sentiment analysis model
tf.reset_default_graph()
if tf.get_default_session():
    tf.get_default_session().close()
sess = tf.Session()
batch_size = 1
lstm_size = 128
# max_len =  100

with tf.variable_scope('imdb', reuse=False):
    model = models.SentimentModel(batch_size=batch_size,
                                  lstm_size=lstm_size,
                                  max_len=max_len,
                                  embeddings_dim=300, vocab_size=dist_mat.shape[1], is_train=False)
saver = tf.train.Saver()
sess_path = 'models/imdb_model'
saver.restore(sess, sess_path)

# Loading the Google Language model
goog_lm = LM()


# ================= ATTACK=======================

with tf.variable_scope('imdb', reuse=True):
    batch_model = models.SentimentModel(batch_size=pop_size,
                                        lstm_size=lstm_size,
                                        max_len=max_len,
                                        embeddings_dim=300, vocab_size=dist_mat.shape[1], is_train=False)

with tf.variable_scope('imdb', reuse=True):
    neighbour_model = models.SentimentModel(batch_size=n1,
                                            lstm_size=lstm_size,
                                            max_len=max_len,
                                            embeddings_dim=300, vocab_size=dist_mat.shape[1], is_train=False)
ga_attacker = GeneticAtack(sess, model, batch_model, neighbour_model, dataset, dist_mat,
                           skip_list,
                           goog_lm, max_iters=max_iters,
                           pop_size=pop_size,

                           n1=n1,
                           n2=K,
                           use_lm=True, use_suffix=False)


test_size = len(dataset.test_y)
test_idx_list = np.arange(len(dataset.test_y))

test_list = []

cur_result_folder = f'{algo}_{dataset_name}/{SEED}'
my_file.create_folder(GA_OUT_PATH, cur_result_folder)
cur_log_file = open(my_file.real_path_of(GA_OUT_PATH, cur_result_folder, 'log.txt'), 'a')
cur_logger = GAIMDBLogger(cur_log_file)
cur_recorder = GARecorderIMDB()

st = time()

for test_idx in test_idx_list:
    x_orig = test_x[test_idx]
    orig_label = test_y[test_idx]
    orig_preds = model.predict(sess, x_orig[np.newaxis, :])[0]

    if np.argmax(orig_preds) != orig_label:
        print('skipping wrong classifed ..')
        print('--------------------------')
        continue
    x_len = np.sum(np.sign(x_orig))
    if x_len < 10:
        print('skipping too short input..')
        print('--------------------------')
        continue
    if x_len >= 100:
        print('skipping too long input..')
        print('--------------------------')
        continue

    test_list.append(test_idx)
    target_label = 1 if orig_label == 0 else 0
    x_adv = ga_attacker.attack(x_orig, target_label)

    cur_logger.log_attack_result(x_adv, x_orig, x_len, ga_attacker)
    cur_recorder.record_result(test_idx, target_label, x_adv, x_orig, x_len)

    if TEST_SIZE and (len(test_list) >= TEST_SIZE):
        break

    cur_t = time()
    print(f'Current use {cur_t - st : .2f}s, average {(cur_t - st) / len(test_list): .2f}s')


parameter_str = f'test size={TEST_SIZE}\npop size = {pop_size}\nmax iteration={max_iters}\nN={n1}\nK={K}'
cur_logger.log_write(parameter_str)

cur_logger.summary()
cur_logger.close_file()

cur_recorder.test_idx_list = test_list
cur_recorder.save_results(my_file.real_path_of(GA_OUT_PATH, cur_result_folder))

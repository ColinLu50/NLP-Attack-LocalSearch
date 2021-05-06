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
from GA_related.shared.attacks_LS import LSAttackIMDB
from GA_related.shared.ga_logger import GAIMDBLogger
from GA_related.shared.ga_recorder import GARecorderIMDB

# from src.ga_reimp.shared import GA_IMDB_ALL_PATH
from utils import my_file

GA_OUT_PATH = 'out/GA_related/'

# tf allow grouth
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)

# ============== Paras =========================
algo = 'EMLM_LS'
dataset_name = 'IMDB'

SEED = 1001

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

# # Demonstrating how we find the most similar words
# for i in range(300, 305):
#     src_word = i
#     nearest, nearest_dist = glove_utils.pick_most_similar_words(src_word, dist_mat, 20, 0.5)
#
#     print('Closest to `%s` are:' % (dataset.inv_dict[src_word]))
#     for w_id, w_dist in zip(nearest, nearest_dist):
#         print(' -- ', dataset.inv_dict[w_id], ' ', w_dist)
#
#     print('----')

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

# # =============== demo ==============
# src_word = dataset.dict['play']
# nearest, nearest_dist = glove_utils.pick_most_similar_words(src_word, dist_mat,20)
# nearest_w = [dataset.inv_dict[x] for x in nearest]
# print('Closest to `%s` are %s' %(dataset.inv_dict[src_word], nearest_w))
# prefix = 'is'
# suffix = 'with'
# lm_preds = goog_lm.get_words_probs(prefix, nearest_w, suffix)
# print('most probable is ', nearest_w[np.argmax(lm_preds)])
# # ============================================

# ================= ATTACK=======================
pop_size = 60
max_iters = 20
n1 = 8
K = 4

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
ga_attacker = LSAttackIMDB(sess, model, batch_model, neighbour_model, dataset, dist_mat,
                           goog_lm,
                           n1=n1,
                           n2=K,
                           use_lm=True, use_suffix=False)

# SAMPLE_SIZE = 5000
# TEST_SIZE = 20
# test_idx = np.random.choice(len(dataset.test_y), SAMPLE_SIZE, replace=False)
# test_len = []
# for i in range(SAMPLE_SIZE):
#     test_len.append(len(dataset.test_seqs2[test_idx[i]]))
# print('Shortest sentence in our test set is %d words' %np.min(test_len))

TEST_SIZE = None
test_size = len(dataset.test_y)
test_idx_list = np.arange(len(dataset.test_y))
# np.random.shuffle(test_idx_list)

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

cur_logger.summary()

cur_recorder.test_idx_list = test_list


cur_recorder.save_results(my_file.real_path_of(GA_OUT_PATH, cur_result_folder))


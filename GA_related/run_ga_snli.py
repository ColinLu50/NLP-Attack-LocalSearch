import os
import sys

sys.path.append(sys.path[0] + '/../')
import tensorflow as tf
import numpy as np
from time import time
from datetime import datetime
import pickle


from GA_related.shared.attacks_GA import EntailmentAttack
from GA_related.snli_rnn_train import build_model

from GA_related.shared.ga_logger import GALoggerSNLI
from GA_related.shared.ga_recorder import GARecorderSNLI
from GA_related.shared.goog_lm import LM

from utils import my_file

# ================= paras ==========================
# 5555, 6666, 7777, 8888, 9999, 1010, 111, 222, 333
SEED = 8888
TEST_SIZE = None

S = 60
G = 20
N = 8
K = 4

algo = 'EMLM_GA'
dataset_name = 'SNLI'
tag = 'all'

np.random.seed(SEED)
tf.set_random_seed(SEED)

# tf allow grouth
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)


with open('./nli_tokenizer.pkl', 'rb') as fh:
    tokenizer = pickle.load(fh)

with open('./nli_testing.pkl', 'rb') as fh:
    test = pickle.load(fh)
# test = my_file.load_pkl_in_repo(GA_SNLI_ALL_PATH, 'nli_testing.pkl')

vocab= {w:i for (w, i) in tokenizer.word_index.items()}
inv_vocab = {i:w for (w,i) in vocab.items()}

def reconstruct(sent, inv_vocab):
    word_list = [inv_vocab[w] for w in sent if w != 0]
    return ' '.join(word_list)

def visulaize_result(model, attack_input, attack_output):
    str_labels = ['Contradiction', 'neutral', 'entailment']
    orig_pred = model.predict(attack_input)
    adv_pred = model.predict([attack_output[0][np.newaxis,:], attack_output[1][np.newaxis,:]])
    print('Original pred = {} ({:.2f})'.format(str_labels[np.argmax(orig_pred[0])], np.max(orig_pred[0])))
    print(reconstruct(attack_input[0].ravel(), inv_vocab) , ' || ', reconstruct(attack_input[1].ravel(), inv_vocab))
    print('-' * 40)
    print('New pred = {} ({:.2f})'.format(str_labels[np.argmax(adv_pred[0])], np.max(adv_pred[0])))
    print(reconstruct(attack_output[0].ravel(), inv_vocab) , ' || ', reconstruct(attack_output[1].ravel(), inv_vocab))


# Building the model

VOCAB = len(tokenizer.word_counts) + 1
LABELS = {'contradiction': 0, 'neutral': 1, 'entailment': 2}

model = build_model()
model_path = 'nli_model.h5'
model.load_weights(model_path)

# Test accuracy
test_accuracy = model.evaluate([test[0], test[1]], test[2])[1]
print('\nTest accuracy = ', test_accuracy)

# Fooling the model
dist_mat_path = 'aux_files/nli_dist_counter_42390.npy'
dist_mat = np.load(dist_mat_path)
skip_words_path = 'aux_files/nli_missed_embeddings_counter_42390.npy'
skip_words = np.load(skip_words_path)

def visulaize_result(model, attack_input, attack_output):
    str_labels = ['Contradiction', 'neutral', 'entailment']
    orig_pred = model.predict(attack_input)
    adv_pred = model.predict([attack_output[0][np.newaxis,:], attack_output[1][np.newaxis,:]])
    print('Original pred = {} ({:.2f})'.format(str_labels[np.argmax(orig_pred[0])], np.max(orig_pred[0])))
    print(reconstruct(attack_input[0].ravel(), inv_vocab) , ' || ', reconstruct(attack_input[1].ravel(), inv_vocab))
    print('-' * 40)
    print('New pred = {} ({:.2f})'.format(str_labels[np.argmax(adv_pred[0])], np.max(adv_pred[0])))
    print(reconstruct(attack_output[0].ravel(), inv_vocab) , ' || ', reconstruct(attack_output[1].ravel(), inv_vocab))







goog_lm = LM()
adversary = EntailmentAttack(model, dist_mat, inv_vocab, goog_lm, pop_size=S, max_iters=G, n1=N, n2=K, use_suffix=False)
# adversary = EntailmentAttack(model, dist_mat, pop_size=128, max_iters=12, n1=5)




# test_idxs = np.random.choice(len(test[0]), size=TEST_SIZE, replace=False)
test_idxs = np.arange(len(test[0]))
np.random.shuffle(test_idxs)

test_list = []

GA_OUT_PATH = 'out/GA_related/'
cur_result_folder = f'{algo}_{dataset_name}/{SEED}'
my_file.create_folder(GA_OUT_PATH, cur_result_folder)
cur_log_file = open(my_file.real_path_of(GA_OUT_PATH, cur_result_folder, 'log.txt'), 'a')
cur_logger = GALoggerSNLI(cur_log_file)
cur_recorder = GARecorderSNLI()

st = time()
for i in range(len(test_idxs)):
    print('----------------------------------')
    test_idx = test_idxs[i]
    attack_input = [test[0][test_idx][np.newaxis,:], test[1][test_idx][np.newaxis,:]]
    x_len = np.sum(np.sign(attack_input[1]))
    if x_len < 10 or x_len > 100:
        print('skip too short')
        continue
    attack_pred = np.argmax(model.predict(attack_input))
    true_label = np.argmax(test[2][test_idx])
    if attack_pred != true_label:
        print('Wrong classified')
    else:
        if true_label == 2:
            target = 0
        elif true_label == 0:
            target = 2
        else:
            target = 0 if np.random.uniform() < 0.5 else 2

        test_list.append(test_idx)
        attack_result = adversary.attack(attack_input, target)

        # check result
        if attack_result is not None:
            success_pred = model.predict([attack_result[0][np.newaxis, :], attack_result[1][np.newaxis, :]])[0]
            assert np.argmax(success_pred) == target

        cur_logger.log_attack_result(attack_result, attack_input, x_len, adversary)
        cur_recorder.record_result(test_idx, target, attack_result, attack_input, x_len)

        cur_t = time()
        print(f'Current use {cur_t - st : .2f}s, average {(cur_t - st) / len(test_list): .2f}s')

        if TEST_SIZE and len(test_list) == TEST_SIZE:
            break


parameter_str = f'SEED={SEED}, Test Size={TEST_SIZE}, S={S}, G={G}, N={N}, K={K}'
cur_logger.summary()
cur_logger.close_file()

cur_recorder.test_idx_list = test_list


cur_recorder.save_results(my_file.real_path_of(GA_OUT_PATH, cur_result_folder))


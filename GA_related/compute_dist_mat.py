import sys

sys.path.append(sys.path[0] + '/../../')

import numpy as np
import tensorflow as tf

# tf allow grouth
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)

from GA_related.shared import glove_utils



MAX_VOCAB_SIZE = 50000
npy_path1 = 'aux_files/embeddings_counter_%d.npy' % (MAX_VOCAB_SIZE)
embedding_matrix = np.load((npy_path1))
npy_path2 = 'aux_files/missed_embeddings_counter_%d.npy' % (MAX_VOCAB_SIZE)
missed = np.load((npy_path2))

c_ = -2 * np.dot(embedding_matrix.T, embedding_matrix)
a = np.sum(np.square(embedding_matrix), axis=0).reshape((1, -1))
b = a.T
dist = a + b + c_

save_path1 = 'aux_files/dist_counter_%d.npy' % (MAX_VOCAB_SIZE)
np.save((save_path1), dist)

# Try an example
dataset = 'aux_files/dataset_%d.pkl' % MAX_VOCAB_SIZE
src_word = dataset.dict['good']
neighbours, neighbours_dist = glove_utils.pick_most_similar_words(src_word, dist)
print('Closest words to `good` are :')
result_words = [dataset.inv_dict[x] for x in neighbours]
print(result_words)

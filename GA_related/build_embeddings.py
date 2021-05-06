import sys

sys.path.append(sys.path[0] + '/../../../')
import os
import pickle

import numpy as np

import tensorflow as tf

from GA_related import data_utils
from GA_related.shared import glove_utils


# tf allow grouth
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)

IMDB_PATH = 'IMDB_datset_path'
MAX_VOCAB_SIZE = 50000
GLOVE_PATH = "GLOVE_FILE_PATH"
COUNTER_FITTED_VECTOR_FILE_PATH = "COUNTER_FITTED_VECTOR_FILE_PATH"


if not os.path.exists('aux_files'):
    os.mkdir('aux_files')
imdb_dataset = data_utils.IMDBDataset(path=IMDB_PATH, max_vocab_size=MAX_VOCAB_SIZE)

# save the dataset
with open(('aux_files/dataset_%d.pkl' % (MAX_VOCAB_SIZE)), 'wb') as f:
    pickle.dump(imdb_dataset, f)
# my_file.save_pkl_in_repo(imdb_dataset, GA_IMDB_ALL_PATH, 'aux_files/dataset_%d.pkl' % (MAX_VOCAB_SIZE))

# create the glove embeddings matrix (used by the classification model)
glove_model = glove_utils.loadGloveModel(GLOVE_PATH)
glove_embeddings, _ = glove_utils.create_embeddings_matrix(glove_model, imdb_dataset.dict, imdb_dataset.full_dict)
# save the glove_embeddings matrix
p1 = 'aux_files/embeddings_glove_%d.npy' % (MAX_VOCAB_SIZE)
np.save(p1, glove_embeddings)

# Load the counterfitted-vectors (used by our attack)
glove2 = glove_utils.loadGloveModel(COUNTER_FITTED_VECTOR_FILE_PATH)
# create embeddings matrix for our vocabulary
counter_embeddings, missed = glove_utils.create_embeddings_matrix(glove2, imdb_dataset.dict, imdb_dataset.full_dict)

# save the embeddings for both words we have found, and words that we missed.
npy_path1 = 'aux_files/embeddings_counter_%d.npy' % (MAX_VOCAB_SIZE)
np.save((npy_path1), counter_embeddings)
npy_path2 = 'aux_files/missed_embeddings_counter_%d.npy' % (MAX_VOCAB_SIZE)
np.save((npy_path2), missed)
print('All done')

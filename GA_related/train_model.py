import sys

sys.path.append(sys.path[0] + '/../')
import os
from keras.preprocessing.sequence import pad_sequences

import numpy as np
import tensorflow as tf

from GA_related import models

# from src.utils import my_file
# from src.ga_reimp.shared import COUNTER_FITTED_VECTOR_FILE_PATH, GA_IMDB_ALL_PATH, GLOVE_FILE_PATH


# tf allow grouth
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)

IMDB_PATH = 'path to/aclImdb'
MAX_VOCAB_SIZE = 50000

if __name__ == '__main__':
    with open(('aux_files/dataset_%d.pkl' %MAX_VOCAB_SIZE), 'rb') as f:
        dataset = pickle.load(f)

    # TODO(malzantot): should we keep using the normal glove embeddings here ?
    p1 = 'aux_files/embeddings_glove_%d.npy' %(MAX_VOCAB_SIZE)
    embedding_matrix = np.load((p1))
    max_len = 250
    train_x = pad_sequences(dataset.train_seqs2, maxlen=max_len, padding='post')
    train_y = np.array(dataset.train_y)
    test_x = pad_sequences(dataset.test_seqs2, maxlen=max_len, padding='post')
    test_y = np.array(dataset.test_y)
    sess = tf.Session()
    batch_size = 64
    lstm_size = 128
    num_epochs = 20
    with tf.variable_scope('imdb', reuse=False):
        model = models.SentimentModel(batch_size=batch_size,
                                      lstm_size = lstm_size,
                                      max_len = max_len,
                                      keep_probs=0.8,
                                      embeddings_dim=300, vocab_size=embedding_matrix.shape[1],
                                      is_train = True)
    with tf.variable_scope('imdb', reuse=True):
        test_model = models.SentimentModel(batch_size=batch_size,
                                           lstm_size = lstm_size,
                                           max_len = max_len, keep_probs=0.8,
                                           embeddings_dim=300, vocab_size=embedding_matrix.shape[1], is_train= False)
    sess.run(tf.global_variables_initializer())
    sess.run(tf.assign(model.embedding_w, embedding_matrix.T))
    print('Training..')
    for i in range(num_epochs):
        epoch_loss, epoch_accuracy = model.train_for_epoch(sess, train_x, train_y)
        print(i, ' ', epoch_loss, ' ', epoch_accuracy)
        print('Train accuracy = ', test_model.evaluate_accuracy(sess, train_x, train_y))
        print('Test accuracy = ', test_model.evaluate_accuracy(sess, test_x, test_y))
    if not os.path.exists('models'):
        os.mkdir('models')
    saver = tf.train.Saver()
    if not os.path.exists('models'):
        os.mkdir('models')
    saver = tf.train.Saver()
    saver.save(sess, 'models/imdb_model')
    print('All done')
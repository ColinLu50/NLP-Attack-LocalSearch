# os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import sys

sys.path.append(sys.path[0] + '/../')
import keras
from keras import backend as K
import tensorflow as tf
import argparse

from sklearn.utils import shuffle

from SbGS_related.config import config
from SbGS_related.read_files import split_imdb_files, split_agnews_files
from SbGS_related.word_level_process import word_process
from SbGS_related.char_level_process import char_process
from SbGS_related.neural_networks import word_cnn, char_cnn, bd_lstm, lstm

from utils import my_file

tf.set_random_seed(222)


tf_config = tf.ConfigProto(allow_soft_placement=True)
tf_config.gpu_options.allow_growth = True
K.set_session(tf.Session(config=tf_config))

parser = argparse.ArgumentParser(
    description='Train a text classifier.')
parser.add_argument('-m', '--model',
                    help='The model of text classifier',
                    choices=['word_cnn', 'char_cnn', 'word_lstm', 'word_bdlstm'],
                    default='word_cnn')
parser.add_argument('-d', '--dataset',
                    help='Data set',
                    choices=['imdb', 'agnews'],
                    default='imdb')
parser.add_argument('-l', '--level',
                    help='The level of process dataset',
                    choices=['word', 'char'],
                    default='word')


def train_text_classifier():
    dataset = args.dataset
    x_train = y_train = x_test = y_test = None
    if dataset == 'imdb':
        train_texts, train_labels, test_texts, test_labels = split_imdb_files()
        if args.level == 'word':
            x_train, y_train, x_test, y_test = word_process(train_texts, train_labels, test_texts, test_labels, dataset)
        elif args.level == 'char':
            x_train, y_train, x_test, y_test = char_process(train_texts, train_labels, test_texts, test_labels, dataset)
    elif dataset == 'agnews':
        train_texts, train_labels, test_texts, test_labels = split_agnews_files()
        if args.level == 'word':
            x_train, y_train, x_test, y_test = word_process(train_texts, train_labels, test_texts, test_labels, dataset)
        elif args.level == 'char':
            x_train, y_train, x_test, y_test = char_process(train_texts, train_labels, test_texts, test_labels, dataset)

    x_train, y_train = shuffle(x_train, y_train, random_state=0)

    # Take a look at the shapes
    print('dataset:', dataset, '; model:', args.model, '; level:', args.level)
    print('X_train:', x_train.shape)
    print('y_train:', y_train.shape)
    print('X_test:', x_test.shape)
    print('y_test:', y_test.shape)

    log_dir = './logs/{}/all_{}/'.format(dataset, args.model)
    tb_callback = keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=0, write_graph=True)

    model_path = r'./runs/{}/{}.dat'.format(dataset, args.model)
    my_file.create_folder(r'./runs/{}'.format(dataset))
    model = batch_size = epochs = None
    assert args.model[:4] == args.level

    if args.model == "word_cnn":
        model = word_cnn(dataset)
        batch_size = config.wordCNN_batch_size[dataset]
        epochs = config.wordCNN_epochs[dataset]
    elif args.model == "word_bdlstm":
        model = bd_lstm(dataset)
        batch_size = config.bdLSTM_batch_size[dataset]
        epochs = config.bdLSTM_epochs[dataset]
    elif args.model == "char_cnn":
        model = char_cnn(dataset)
        batch_size = config.charCNN_batch_size[dataset]
        epochs = config.charCNN_epochs[dataset]
    elif args.model == "word_lstm":
        model = lstm(dataset)
        batch_size = config.LSTM_batch_size[dataset]
        epochs = config.LSTM_epochs[dataset]

    print('Train...')
    print('batch_size: ', batch_size, "; epochs: ", epochs)
    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              validation_split=0.2,
              shuffle=True,
              callbacks=[tb_callback])
    scores = model.evaluate(x_test, y_test)
    print('test_loss: %f, accuracy: %f' % (scores[0], scores[1]))
    print('Saving model weights...')
    model.save_weights(model_path)


if __name__ == '__main__':
    args = parser.parse_args()
    train_text_classifier()
import argparse

from SbGS_related.char_level_process import char_process
from SbGS_related.neural_networks import word_cnn, bd_lstm, char_cnn, lstm
from SbGS_related.read_files import split_imdb_files, split_agnews_files
from SbGS_related.word_level_process import word_process

# from src.utils import my_file


def get_parser():
    parser = argparse.ArgumentParser(
        description='Craft adversarial examples for a text classifier.')
    parser.add_argument('--clean_samples_cap',
                        help='Amount of clean(test) samples to fool',
                        type=int, default=None)
    parser.add_argument('-m', '--model',
                        help='The model of text classifier',
                        choices=['word_cnn', 'char_cnn', 'word_lstm', 'word_bdlstm'],
                        default='word_cnn')
    parser.add_argument('-d', '--dataset',
                        help='Data set',
                        choices=['imdb', 'agnews', 'yahoo'],
                        default='imdb')
    parser.add_argument('-l', '--level',
                        help='The level of process dataset',
                        choices=['word', 'char'],
                        default='word')

    return parser


def load_dataset(dataset, level):
    x_test = y_test = None
    test_texts = None

    if dataset == 'imdb':
        train_texts, train_labels, test_texts, test_labels = split_imdb_files()
        if level == 'word':
            x_train, y_train, x_test, y_test = word_process(train_texts, train_labels, test_texts, test_labels, dataset)
        elif level == 'char':
            x_train, y_train, x_test, y_test = char_process(train_texts, train_labels, test_texts, test_labels, dataset)
    elif dataset == 'agnews':
        train_texts, train_labels, test_texts, test_labels = split_agnews_files()
        if level == 'word':
            x_train, y_train, x_test, y_test = word_process(train_texts, train_labels, test_texts, test_labels, dataset)
        elif level == 'char':
            x_train, y_train, x_test, y_test = char_process(train_texts, train_labels, test_texts, test_labels, dataset)

    return train_texts, train_labels, test_texts, test_labels, x_train, y_train, x_test, y_test


def load_model(args):
    assert args.model[:4] == args.level
    model = None
    if args.model == "word_cnn":
        model = word_cnn(args.dataset)
    elif args.model == "word_bdlstm":
        model = bd_lstm(args.dataset)
    elif args.model == "char_cnn":
        model = char_cnn(args.dataset)
    elif args.model == "word_lstm":
        model = lstm(args.dataset)

    model_path = './runs/{}/{}.dat'.format(args.dataset, args.model)
    model.load_weights(model_path)
    print('model path:', model_path)

    return model
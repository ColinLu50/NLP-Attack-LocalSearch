import os
import re
import sys
import csv
import numpy as np
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical

from SbGS_related.config import config
# from src.utils import my_file

IMDB_data_folder = 'data_set/aclImdb/'
AGNews_data_folder = 'data_set/ag_news_csv/'



def rm_tags(text):
    re_tag = re.compile(r'<[^>]+>')
    return re_tag.sub('', text)


def read_imdb_files(filetype):
    """
    filetype: 'train' or 'test'
    """

    # [0,1] means positive，[1,0] means negative
    all_labels = []
    for _ in range(12500):
        all_labels.append([0, 1])
    for _ in range(12500):
        all_labels.append([1, 0])

    all_texts = []
    file_list = []
    path = IMDB_data_folder
    pos_path = os.path.join(path, filetype, 'pos/')
    for file in os.listdir(pos_path):
        file_list.append(pos_path + file)
    neg_path = os.path.join(path, filetype, 'neg/')
    for file in os.listdir(neg_path):
        file_list.append(neg_path + file)
    for file_name in file_list:
        with open(file_name, 'r', encoding='utf-8') as f:
            all_texts.append(rm_tags(" ".join(f.readlines())))
    return all_texts, all_labels


def split_imdb_files():
    print('Processing IMDB dataset')
    train_texts, train_labels = read_imdb_files('train')
    test_texts, test_labels = read_imdb_files('test')
    return train_texts, train_labels, test_texts, test_labels



def read_agnews_files(filetype):
    texts = []
    labels_index = []  # The index of label of all input sentences, which takes the values 1,2,3,4
    doc_count = 0  # number of input sentences
    path = os.path.join(AGNews_data_folder, r'{}.csv'.format(filetype))
    csvfile = open(path, 'r')
    for line in csv.reader(csvfile, delimiter=',', quotechar='"'):
        content = line[1] + ". " + line[2]
        texts.append(content)
        labels_index.append(line[0])
        doc_count += 1

    # Start document processing
    labels = []
    for i in range(doc_count):
        label_class = np.zeros(config.num_classes['agnews'], dtype='float32')
        label_class[int(labels_index[i]) - 1] = 1
        labels.append(label_class)

    return texts, labels, labels_index


def split_agnews_files():
    print("Processing AG's News dataset")
    train_texts, train_labels, _ = read_agnews_files('train')  # 120000
    test_texts, test_labels, _ = read_agnews_files('test')  # 7600
    return train_texts, train_labels, test_texts, test_labels


if __name__ == '__main__':
    split_agnews_files()
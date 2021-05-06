# coding: utf-8
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import sys
sys.path.append(sys.path[0] + '/../../')
import os

import numpy as np
import tensorflow as tf
from keras import backend as K
import time
import spacy

from SbGS_related.word_level_process import get_tokenizer
from SbGS_related.PWWS.adversarial_tools import adversarial_paraphrase
from SbGS_related.model_wrapper import ForwardGradWrapper
from SbGS_related.unbuffered import Unbuffered

from SbGS_related.arg_utils import get_parser, load_dataset, load_model
from SbGS_related.PWWS_logger import PWWSLogger
from SbGS_related.PWWS_recorder import PWWSRecorder
from SbGS_related.PWWS_paras import PWWS_OUT_PATH
# from src.utils import my_file



sys.stdout = Unbuffered(sys.stdout)
config = tf.ConfigProto(allow_soft_placement=True)
config.gpu_options.allow_growth = True
K.set_session(tf.Session(config=config))
np.random.seed(2333)

# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# embedding_tool = spacy.load('en', tagger=False, entity=False)
embedding_tool = spacy.load('en_core_web_sm')


parser = get_parser()


def fool_text_classifier():
    algo = 'PWWS'
    dataset = args.dataset
    model_name = args.model

    result_folder = f'{algo}_{dataset}/{model_name}'

    dataset = args.dataset

    # get tokenizer
    tokenizer = get_tokenizer(dataset)

    # Read data set
    train_texts, train_labels, test_texts, test_labels, x_train, y_train, x_test, y_test = load_dataset(dataset, args.level)

    # correct the number of test size
    clean_samples_cap = args.clean_samples_cap
    if clean_samples_cap is None:
        clean_samples_cap = len(test_texts)
    print('clean_samples_cap:', clean_samples_cap)

    model = load_model(args)

    # evaluate classification accuracy of model on clean samples
    scores_origin = model.evaluate(x_test[:clean_samples_cap], y_test[:clean_samples_cap])
    print('clean samples origin test_loss: %f, accuracy: %f' % (scores_origin[0], scores_origin[1]))
    all_scores_origin = model.evaluate(x_test, y_test)
    print('all origin test_loss: %f, accuracy: %f' % (all_scores_origin[0], all_scores_origin[1]))

    grad_guide = ForwardGradWrapper(model)

    # classes_prediction = grad_guide.predict_classes(x_test[: clean_samples_cap])
    classes_prediction = grad_guide.predict_classes(x_test)
    print('Crafting adversarial examples...')

    start_t = time.time()

    # logger
    cur_logger = PWWSLogger(result_folder)

    # recorder
    cur_recorder = PWWSRecorder()

    attack_num = 0

    test_size = len(test_texts)
    test_indices = np.arange(test_size)

    doc_len_restrict = True
    if dataset == 'yahoo':
        doc_len_restrict = False
        np.random.shuffle(test_indices)
        test_indices = test_indices[:test_size // 3]

    print('test size:', len(test_indices))

    for index in test_indices:
        text = test_texts[index]
        if np.argmax(y_test[index]) != classes_prediction[index]:  # wrong classified
            continue

        doc = embedding_tool(text)
        doc_len = len(doc)
        if doc_len_restrict and (doc_len > 100 or doc_len < 10):
            print(f'text length {doc_len} out of bounds')
            continue
        print('=============== attack', index, '==================')
        attack_num += 1
        # If the ground_true label is the same as the predicted label
        grad_guide.start_attack()
        adv_doc, adv_y, modi_rate, _no_use_NE_rate, _no_use_change_tuple_list = adversarial_paraphrase(input_text=text,
                                                                                      true_y=np.argmax(y_test[index]),
                                                                                      grad_guide=grad_guide,
                                                                                      tokenizer=tokenizer,
                                                                                      dataset=dataset,
                                                                                      level=args.level)
        query_num = grad_guide.end_attack() // 2 + 1 # only record valid query number

        attack_success = adv_y != np.argmax(y_test[index])

        cur_logger.log_attack_result(attack_success, modi_rate, query_num)
        cur_recorder.record_result(index, attack_success, adv_doc, adv_y, modi_rate)


        if (clean_samples_cap is not None) and (attack_num > clean_samples_cap):
            break


        cur_t = time.time()
        print(f'current use:{cur_t - start_t}s avg {(cur_t - start_t) / attack_num}s')
        print('=============== attack end ===================')


    cur_logger.summary()
    cur_logger.close_file()

    cur_recorder.save_results(os.path.join(PWWS_OUT_PATH, result_folder))


if __name__ == '__main__':
    args = parser.parse_args()
    fool_text_classifier()
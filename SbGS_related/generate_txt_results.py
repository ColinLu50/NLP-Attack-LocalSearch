import sys
sys.path.append(sys.path[0] + '/../')
import os

from tqdm import tqdm
import numpy as np
import spacy

from SbGS_related.arg_utils import load_dataset
from SbGS_related.PWWS_paras import PWWS_OUT_PATH

from utils import my_file


# ====================== read results ==============================
search_method = 'LS' # PWWS, LS
dataset_name = 'agnews'  # ['imdb', 'agnews']
model_name = 'char_cnn' #['word_cnn', 'char_cnn', 'word_bdlstm']
level = model_name[:4]

result_folder_path = os.path.join(PWWS_OUT_PATH, f'/{search_method}_{dataset_name}/{model_name}')

def read_from_folder(result_folder):
    success_idx_list, success_x_list, success_y_list =\
        my_file.load_pkl_in_repo(result_folder, './success.pkl')

    return success_idx_list, success_x_list, success_y_list



success_idx_list, success_x_list, success_y_list = read_from_folder(result_folder_path)

# read raw data
_, _, test_texts, test_labels, _, _, x_test, y_test = load_dataset(dataset_name, level)


# save
# open save file
orig_plain_text_filename = 'orig.txt'
adv_plain_text_filename = 'adv.txt'

orig_txtfile = open(my_file.real_path_of(result_folder_path, orig_plain_text_filename), 'w')
adv_txtfile = open(my_file.real_path_of(result_folder_path, adv_plain_text_filename), 'w')

nlp = spacy.load('en_core_web_sm')

for i, test_idx in tqdm(enumerate(success_idx_list)):
    orig_y = np.argmax(test_labels[test_idx])

    # process original text to keep the same format
    doc_ = nlp(orig_text)
    std_tokens = []
    for position, token in enumerate(doc_):
        word = token.text
        if word.replace(" ", "") == "":
            continue
        std_tokens.append(word)
    orig_text = ' '.join(std_tokens)

    adv_text = success_x_list[i]
    adv_y = success_y_list[i]

    assert adv_y != orig_y

    orig_txtfile.write(orig_text + '\n')
    adv_txtfile.write(adv_text + '\n')





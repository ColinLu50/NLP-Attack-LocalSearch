import os
import sys
sys.path.append(sys.path[0] + '/../')

import numpy as np
import pickle

from utils import my_file


result_folder = ''

with open('./nli_tokenizer.pkl', 'rb') as fh:
    tokenizer = pickle.load(fh)

# vocab= {w:i for (w, i) in tokenizer.word_index.items()}
inv_vocab = {i:w for (w, i) in tokenizer.word_index.items()}
def reconstruct(sent):
    word_list = [inv_vocab[w] for w in sent if w != 0]
    return ' '.join(word_list)

# load dataset
with open('./nli_testing.pkl', 'rb') as fh:
    test = pickle.load(fh)

# load result

test_idx_list = my_file.load_pkl(os.path.join(result_folder, 'test_list.pkl'))
success_idx_list, success_target_list, success_x_list = my_file.load_pkl(os.path.join(result_folder, 'success.pkl'))
# self.long_fail_idx_list, self.long_fail_target_list, long_fail_x_list

orig_plain_text_filename = 'orig.txt'
adv_plain_text_filename = 'adv.txt'

orig_txtfile = open(os.path.join(result_folder, orig_plain_text_filename), 'w')
adv_txtfile = open(os.path.join(result_folder, adv_plain_text_filename), 'w')


for i, test_idx in enumerate(success_idx_list):
    # only read hypothesis
    orig_x1 = test[1][test_idx]
    orig_y = np.argmax(test[2][test_idx])

    adv_x1 = success_x_list[i][1]
    adv_y = success_target_list[i]

    assert orig_y != adv_y

    orig_text = reconstruct(orig_x1)
    adv_text = reconstruct(adv_x1)


    # write to text file
    orig_txtfile.write(orig_text + '\n')
    adv_txtfile.write(adv_text + '\n')










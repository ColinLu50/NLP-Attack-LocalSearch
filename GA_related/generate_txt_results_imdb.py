import os
import sys
sys.path.append(sys.path[0] + '/../')

import numpy as np

from utils import my_file

# load result
result_folder = 'path of result'


# load dataset
VOCAB_SIZE = 50000
dataset = 'aux_files/dataset_%d.pkl' % VOCAB_SIZE
orig_test_x = dataset.test_seqs2
orig_test_y = dataset.test_y

# define reconstruct text
def reconstruct_text(x_):
    x_len = np.sum(np.sign(x_))
    x_list = list(x_[:x_len])
    text = dataset.build_text(x_list)
    return text




# test_idx_list = my_file.load_pkl_in_repo(result_folder, 'test_list.pkl')
success_idx_list, success_target_list, success_x_list = my_file.load_pkl(os.path.join(result_folder, 'success.pkl'))
# self.long_fail_idx_list, self.long_fail_target_list, long_fail_x_list

orig_plain_text_filename = 'orig.txt'
adv_plain_text_filename = 'adv.txt'

orig_txtfile = open(os.path.join(result_folder, orig_plain_text_filename), 'w')
adv_txtfile = open(os.path.join(result_folder, adv_plain_text_filename), 'w')


for i, test_idx in enumerate(success_idx_list):
    # only read hypothesis
    orig_x1 = orig_test_x[test_idx]
    orig_y = orig_test_y[test_idx]

    adv_x1 = success_x_list[i]
    adv_y = success_target_list[i]

    # print(orig_y, adv_y)
    assert orig_y != adv_y

    orig_text = reconstruct_text(orig_x1)
    adv_text = reconstruct_text(adv_x1)


    # print(orig_text)
    # print(adv_text)
    #
    # print('\n\n')

    # write to text file
    orig_txtfile.write(orig_text + '\n')
    adv_txtfile.write(adv_text + '\n')

orig_txtfile.close()
adv_txtfile.close()










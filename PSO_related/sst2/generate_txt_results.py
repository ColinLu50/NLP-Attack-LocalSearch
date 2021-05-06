import sys
sys.path.append(sys.path[0] + '/../../')
import os

from tqdm import tqdm
from utils import my_file

# ==================== paras ========================
dataset_path = '/home/workspace/nlp_attack/data/pso_raw/SST_used_data/aux_files/dataset_13837.pkl'

# set the result folder path
folder_path = '/home/workspace/nlp_attack_ls_final/out/pso_related/SST2_BERT_LS_search/3333'

# ============ read dataset ========================

dataset = my_file.load_pkl(dataset_path)
all_test_num = len(dataset.test_y)

# ============ read attack result ================
success_test_idx_list, success_target_list, success_eg_list = \
    my_file.load_pkl(os.path.join(folder_path, 'success_all.pkl'))

# open save file
orig_plain_text_filename = 'orig.txt'
adv_plain_text_filename = 'adv.txt'

orig_txtfile = open(os.path.join(folder_path, orig_plain_text_filename), 'w')
adv_txtfile = open(os.path.join(folder_path, adv_plain_text_filename), 'w')

# for only attack success
for i, success_test_idx in tqdm(enumerate(success_test_idx_list)):
    x_orig = dataset.test_seqs2[success_test_idx]
    y_orig = dataset.test_y[success_test_idx]
    word_list_orig = [dataset.inv_full_dict[token_idx] for token_idx in x_orig if token_idx > 0]
    text_orig = " ".join(word_list_orig)

    x_adv = success_eg_list[i]
    y_adv = success_target_list[i]
    word_list_adv = [dataset.inv_full_dict[token_idx] for token_idx in x_adv if token_idx > 0]
    text_adv = " ".join(word_list_adv)

    assert y_orig != y_adv

    # print('raw:\n', text_adv)
    # print('attack:\n', text_adv)
    #
    # break

    # write to text file
    orig_txtfile.write(text_orig + '\n')
    adv_txtfile.write(text_adv + '\n')


adv_txtfile.close()
orig_txtfile.close()
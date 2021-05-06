import sys
sys.path.append(sys.path[0] + '/../../')
import os

from tqdm import tqdm
from utils import my_file

# ======================== paras=========================================
result_folder = '/home/workspace/nlp_attack_ls_final/out/pso_related/SNLI_BiLSTM_LS_search/3333'
SNLI_DATA_PATH = '/home/workspace/nlp_attack/data/pso_raw/SNLI_used_data'

# ============ read dataset ========================
train, valid, test = my_file.load_pkl(os.path.join(SNLI_DATA_PATH, 'all_seqs.pkl'))
test_s1 = [t[1:-1] for t in test['s1']]
test_s2 = [t[1:-1] for t in test['s2']]

# read tokenizer
tokenizer = my_file.load_pkl(os.path.join(SNLI_DATA_PATH, 'nli_tokenizer.pkl'))
inv_vocab = {i: w for (w, i) in tokenizer.word_index.items()}

def reconstruct_text(x_):
    word_list = []
    for w_idx in x_:
        word_list.append(inv_vocab[w_idx])

    return " ".join(word_list)


# ============ read attack result ================
test_idx_list = my_file.load_pkl(os.path.join(result_folder, 'test_id_list.pkl'))
success_test_idx_list, success_target_list, success_eg_list = \
    my_file.load_pkl(os.path.join(result_folder, 'success_all.pkl'))

# long_fail_test_idx_list, long_fail_target_list, long_fail_eg_list = \
#     my_file.load_pkl_in_repo(result_folder, 'long_fail_all.pkl')

# open save file
orig_plain_text_filename = 'orig.txt'
adv_plain_text_filename = 'adv.txt'

orig_txtfile = open(os.path.join(result_folder, orig_plain_text_filename), 'w')
adv_txtfile = open(os.path.join(result_folder, adv_plain_text_filename), 'w')

# for only attack success
for i, success_test_idx in tqdm(enumerate(success_test_idx_list)):
    # premise_ = test_s1[success_test_idx]
    x2_orig = test_s2[success_test_idx]
    y_orig = test['label'][success_test_idx]
    text2_orig = reconstruct_text(x2_orig)

    x2_adv = success_eg_list[i]
    y_adv = success_target_list[i]
    text2_adv = reconstruct_text(x2_adv)

    assert y_orig != y_adv

    # write to text file
    orig_txtfile.write(text2_orig + '\n')
    adv_txtfile.write(text2_adv + '\n')


adv_txtfile.close()
orig_txtfile.close()
from PSO_related.all_shared.shared_parameters import SUCCESS_THRESHOLD
from utils import my_file


class ResultSaver:

    def __init__(self):
        self.test_idx_list = []

        self.success_test_idx_list = []
        self.success_target_list = []
        self.success_eg_list = []

        self.long_fail_test_idx_list = []
        self.long_fail_target_list = []
        self.long_fail_eg_list = []

    def record_attack(self, attack_result, orig_text, x_len, test_idx, target_label):
        self.test_idx_list.append(test_idx)

        if attack_result is not None:
            num_changes = 0
            for i in range(len(orig_text)):
                if orig_text[i] != attack_result[i]:
                    num_changes += 1
            modify_ratio = num_changes / x_len
            if modify_ratio > SUCCESS_THRESHOLD:
                self.long_fail_test_idx_list.append(test_idx)
                self.long_fail_target_list.append(target_label)
                self.long_fail_eg_list.append(attack_result)
            else:
                self.success_test_idx_list.append(test_idx)
                self.success_eg_list.append(attack_result)
                self.success_target_list.append(target_label)

    def save_to_folder(self, folder_path):
        my_file.save_pkl_in_repo(self.test_idx_list, folder_path, 'test_id_list.pkl')
        my_file.save_pkl_in_repo((self.success_test_idx_list, self.success_target_list, self.success_eg_list),
                                 folder_path,
                                 'success_all.pkl')
        my_file.save_pkl_in_repo((self.long_fail_test_idx_list, self.long_fail_target_list, self.long_fail_eg_list),
                                 folder_path,
                                 'long_fail_all.pkl')

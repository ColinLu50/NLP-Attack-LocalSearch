import os
import numpy as np

from GA_related.shared.paras import SUCCESS_THRESHOLD
from utils import my_file


class GARecorder:

    def __init__(self):
        self.test_idx_list = []

        self.success_idx_list = []
        self.success_x_list = []
        self.success_target_list = []

        self.long_fail_idx_list = []
        self.long_fail_x_list = []
        self.long_fail_target_list = []

    def cal_change_num(self, attack_result, attack_input):
        raise NotImplemented

    def record_result(self, real_idx, target, attack_result, attack_input, x_len):
        if attack_result is not None:
            # change ratio
            num_changes = self.cal_change_num(attack_result, attack_input)
            change_ratio = num_changes / x_len

            if change_ratio > SUCCESS_THRESHOLD:
                self.long_fail_idx_list.append(real_idx)
                self.long_fail_x_list.append(attack_result)
                self.long_fail_target_list.append(target)
            else:
                self.success_idx_list.append(real_idx)
                self.success_x_list.append(attack_result)
                self.success_target_list.append(target)

    def save_results(self, folder_path):

        my_file.create_folder(folder_path)

        my_file.save_pkl(self.test_idx_list, os.path.join(folder_path, './test_list.pkl'))
        my_file.save_pkl((self.success_idx_list, self.success_target_list, self.success_x_list),
                         os.path.join(folder_path, './success.pkl'))
        my_file.save_pkl(
            (self.long_fail_idx_list, self.long_fail_target_list, self.long_fail_x_list),
            os.path.join(folder_path, './long_fail.pkl')
        )


class GARecorderIMDB(GARecorder):

    def cal_change_num(self, attack_result, attack_input):
        num_changes = np.sum(attack_result != attack_input)
        return num_changes


class GARecorderSNLI(GARecorder):
    def cal_change_num(self, attack_result, attack_input):
        h1 = attack_input[1].reshape([-1])
        h2 = attack_result[1].reshape([-1])
        num_changes = np.sum(h1 != h2)
        return num_changes

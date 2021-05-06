from SbGS_related.PWWS_paras import SUCCESS_THRESHOLD
from utils import my_file


class PWWSRecorder:

    def __init__(self):
        self.test_idx_list = []

        self.success_idx_list = []
        self.success_x_list = []
        self.success_y_list = []

        self.long_fail_idx_list = []
        self.long_fail_x_list = []
        self.long_fail_y_list = []

    def record_result(self, test_idx, attack_success, attack_result, y, change_ratio):
        if attack_success:
            if change_ratio > SUCCESS_THRESHOLD:
                self.long_fail_idx_list.append(test_idx)
                self.long_fail_x_list.append(attack_result)
                self.long_fail_y_list.append(y)
            else:
                self.success_idx_list.append(test_idx)
                self.success_x_list.append(attack_result)
                self.success_y_list.append(y)

    def save_results(self, folder_path):

        my_file.save_pkl_in_repo(self.test_idx_list, folder_path, './test_list.pkl')
        my_file.save_pkl_in_repo((self.success_idx_list, self.success_x_list, self.success_y_list), folder_path,
                                 './success.pkl')
        my_file.save_pkl_in_repo(
            (self.long_fail_idx_list, self.long_fail_x_list, self.long_fail_y_list), folder_path,
            './long_fail.pkl')

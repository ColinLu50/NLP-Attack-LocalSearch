import sys

import numpy as np

from SbGS_related.PWWS_paras import SUCCESS_THRESHOLD, PWWS_OUT_PATH
from utils import my_file


class PWWSLogger:

    def __init__(self, result_folder_in_repo, log_file_path=None):

        self.success_count = 0
        self.test_count = 0
        self.long_fail_count = 0

        if result_folder_in_repo is not None:
            my_file.create_folder(PWWS_OUT_PATH, result_folder_in_repo)
            self.log_file = open(my_file.real_path_of(PWWS_OUT_PATH, result_folder_in_repo, 'log.txt'), 'w')
        elif log_file_path is not None:
            self.log_file = open(log_file_path, 'w')

        self.query_num_list = []
        self.success_query_num_list = []
        self.all_success_change_ratio_list = []
        self.change_ratio_list = []

    def cal_change_num(self, attack_result, attack_input):
        raise NotImplemented

    def _flush(self):
        if self.log_file is not None:
            self.log_file.flush()

        sys.stdout.flush()

    def log_write(self, write_str):
        if self.log_file is not None:
            self.log_file.write(write_str + '\n')

    def log_attack_result(self, is_attack_success, change_rate, query_num):
        str1 = f'============== Attack {self.test_count} Results =================='
        # print(str1)
        self.log_write(str1)

        self.test_count += 1
        self.query_num_list.append(query_num)

        if not is_attack_success:
            str1 = f'\tFail'
            self.log_write(str1)
        else:
            str1 = f'\tModification Rate: {change_rate:.2%}'
            self.log_write(str1)
            self.all_success_change_ratio_list.append(change_rate)

            if change_rate > SUCCESS_THRESHOLD:
                self.long_fail_count += 1
                str1 = f'\tFail: Change Rate too High {change_rate:.2%}'
                # print(str1)
                self.log_write(str1)
            else:
                self.success_count += 1
                self.change_ratio_list.append(change_rate)
                self.success_query_num_list.append(query_num)
                str1 = '\tSuccess'
                # print(str1)
                self.log_write(str1)
            # self.success_count += 1
            # self.change_ratio_list.append(change_rate)
            # self.success_query_num_list.append(query_num)
            # str1 = '\tSuccess'
            # # print(str1)
            # self.log_write(str1)

        str1 = f'\tSuccess rate: {(self.success_count / self.test_count): .2%}\n'
        # str2 = f'\t raw : {orig_text}\n\tAttack: {adv_text}\n'
        # print(str1)
        # print(str2)
        self.log_write(str1)
        # self.log_write(str2)

        self._flush()


    def summary(self):
        str_sum = '====================== Summary ===========================\n' + \
                  f'Success rate: {self.success_count} / {self.test_count} = {(self.success_count / self.test_count): .2%}\n' + \
                  f'Modification Rate: {np.mean(self.change_ratio_list): .2%}\n' + \
                  f'All Success Modification Rate {np.mean(self.all_success_change_ratio_list): .2%}\n' +\
                  f'Query Number: {np.mean(self.query_num_list)}\n' + \
                  f'Success Query Number: {np.mean(self.success_query_num_list)}\n' + \
                  f'Long fail number: {self.long_fail_count}, Real success rate {(self.success_count + self.long_fail_count) / self.test_count: .2%}' + \
                  '\n\n'
        print(str_sum)
        self.log_write(str_sum)
        self._flush()

    def close_file(self):
        if self.log_file is not None:
            self.log_file.close()


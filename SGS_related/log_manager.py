import sys

import numpy as np

from utils import my_file
from SGS_related.shared_params import RESULT_FOLDER, SUCCESS_THRESHOLD

class QALogManager:

    def __init__(self, result_folder_in_repo, log_filename='log.txt'):

        # record all
        self.success_count = 0
        self.test_count = 0
        self.long_fail_count = 0
        self.query_num_list = []
        self.success_query_num_list = []
        self.real_success_modif_rate_list = []
        self.modif_rate_list = []

        if result_folder_in_repo is not None:
            my_file.create_folder(RESULT_FOLDER, result_folder_in_repo)
            self.log_file = open(my_file.real_path_of(RESULT_FOLDER, result_folder_in_repo, log_filename), 'w')



    def _flush(self):
        if self.log_file is not None:
            self.log_file.flush()

        sys.stdout.flush()

    def log_write(self, write_str):
        if self.log_file is not None:
            self.log_file.write(write_str + '\n')

    def log_attack_result(self, is_attack_success, modif_rate, query_num):
        str1 = f'============== Attack {self.test_count} Results =================='
        # print(str1)
        self.log_write(str1)

        self.test_count += 1
        self.query_num_list.append(query_num)

        if not is_attack_success:
            str1 = f'\tFail'
            self.log_write(str1)
        else:
            str1 = f'\tModification Rate: {modif_rate:.2%}'
            self.log_write(str1)
            self.real_success_modif_rate_list.append(modif_rate)

            if modif_rate > SUCCESS_THRESHOLD:
                self.long_fail_count += 1
                str1 = f'\tFail: Change Rate too High {modif_rate:.2%}'
                # print(str1)
                self.log_write(str1)
            else:
                self.success_count += 1
                self.modif_rate_list.append(modif_rate)
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
                  f'All number {self.test_count}\n' +\
                  f'Success rate: {self.success_count} / {self.test_count} = {self.success_rate: .2%}\n' + \
                  f'Modification Rate: {self.modification_rate: .2%}\n' + \
                  f'Real Success Modification Rate {self.success_modification_rate: .2%}\n' + \
                  f'Query Number: {self.average_query_num}\n' + \
                  f'Success Query Number: {self.success_average_query_num}\n' + \
                  f'Long fail number: {self.long_fail_count}, Real success rate {self.real_success_rate: .2%}' + \
                  '\n\n'
        print(str_sum)
        self.log_write(str_sum)
        self._flush()

    def close_file(self):
        if self.log_file is not None:
            self.log_file.close()

    @property
    def average_query_num(self):
        if len(self.query_num_list) > 0:
            return np.mean(self.query_num_list)
        else:
            return 0.0

    @property
    def success_average_query_num(self):
        if len(self.success_query_num_list) > 0:
            return np.mean(self.success_query_num_list)
        else:
            return 0.0

    @property
    def success_rate(self):
        if self.test_count > 0:
            return self.success_count / self.test_count
        else:
            return 0.0

    @property
    def real_success_rate(self):
        if self.test_count > 0:
            return (self.success_count + self.long_fail_count) / self.test_count
        else:
            return 0.0

    @property
    def modification_rate(self):
        if len(self.modif_rate_list) > 0:
            return np.mean(self.modif_rate_list)
        else:
            return 0.0

    @property
    def success_modification_rate(self):
        if len(self.real_success_modif_rate_list) > 0:
            return np.mean(self.real_success_modif_rate_list)
        else:
            return 0.0
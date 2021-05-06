import sys

import numpy as np

from PSO_related.all_shared.shared_parameters import SUCCESS_THRESHOLD


class MyLogger:

    def __init__(self, log_file):
        self.log_file = log_file

        self.test_num = 0
        self.succss_num = 0
        self.long_fail_num = 0
        self.modifi_rate_list = []
        self.succes_query_num_list = []
        self.succes_uncached_query_num_list = []
        self.query_num_list = []
        self.uncached_query_num_list = []

    def log_result(self, attack_result, orig_text, x_len, attacker):
        self.test_num += 1
        print('====================== Start', self.test_num, ' ====================')
        self.log_file.write(f'Test {self.test_num}: \n')

        self.query_num_list.append(attacker.query_num)
        self.uncached_query_num_list.append(attacker.query_num - attacker.query_num_cached)

        if attack_result is None:
            print('%d failed' % self.test_num)
            self.log_file.write('\tFail\n')
        else:
            num_changes = 0
            for i in range(len(orig_text)):
                if orig_text[i] != attack_result[i]:
                    num_changes += 1
            print(f'{num_changes} words changed')
            modify_ratio = num_changes / x_len
            if modify_ratio > SUCCESS_THRESHOLD:
                print('Fail: Modification too large', modify_ratio)
                self.long_fail_num += 1
                self.log_file.write(f'\tFail: Modification {modify_ratio:.2%} too large\n')
            else:
                print('Success!')
                self.succss_num += 1
                self.log_file.write('\tSuccess\n')
                self.succes_query_num_list.append(attacker.query_num)
                self.succes_uncached_query_num_list.append(attacker.query_num - attacker.query_num_cached)
                self.modifi_rate_list.append(modify_ratio)

        str1 = f'Attack Success Rate {self.succss_num} / {self.test_num} = {self.succss_num / self.test_num: .2%}'
        print(str1)
        print('============================= END ==============================')
        self.log_file.flush()
        sys.stdout.flush()

    def summary(self):
        result_str = '\n\n================== Summary ========================\n' + \
                     f'Attack Success Rate {self.succss_num} / {self.test_num} = {self.succss_num / self.test_num: .2%}\n' + \
                     f'Mean Modification Rate: {np.mean(self.modifi_rate_list) :.2%}\n' + \
                     f'{self.long_fail_num} attacks fail because it\'s too long, real success rate {(self.succss_num + self.long_fail_num) / self.test_num: .2%}\n' + \
                     f'Average Query Num {np.mean(self.query_num_list)}\n' + \
                     f'Success Average Query Num {np.mean(self.succes_query_num_list)}\n' + \
                     '\n\n'

        print(result_str)
        self.log_file.write(result_str)

        self.log_file.flush()
        sys.stdout.flush()

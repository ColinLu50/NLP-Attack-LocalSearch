import sys

import numpy as np

from GA_related.shared.paras import SUCCESS_THRESHOLD


# def visulaize_result(model, attack_input, attack_output):
#     str_labels = ['Contradiction', 'neutral', 'entailment']
#     orig_pred = model.predict(attack_input)
#     adv_pred = model.predict([attack_output[0][np.newaxis,:], attack_output[1][np.newaxis,:]])
#     print('Original pred = {} ({:.2f})'.format(str_labels[np.argmax(orig_pred[0])], np.max(orig_pred[0])))
#     print(reconstruct(attack_input[0].ravel(), inv_vocab) , ' || ', reconstruct(attack_input[1].ravel(), inv_vocab))
#     print('-' * 40)
#     print('New pred = {} ({:.2f})'.format(str_labels[np.argmax(adv_pred[0])], np.max(adv_pred[0])))
#     print(reconstruct(attack_output[0].ravel(), inv_vocab) , ' || ', reconstruct(attack_output[1].ravel(), inv_vocab))



class GALogger:

    def __init__(self, log_file):

        self.success_count = 0
        self.test_count = 0
        self.long_fail_count = 0

        self.log_file = log_file

        self.query_num_list = []
        self.lm_query_num_list = []
        self.success_query_num_list = []
        self.success_lm_query_num_list = []
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

    def log_attack_result(self, attack_result, attack_input, x_len, attacker):
        str1 = f'============== Attack {self.test_count} Results =================='
        # print(str1)
        self.log_write(str1)

        self.test_count += 1
        self.query_num_list.append(attacker.query_num)
        self.lm_query_num_list.append(attacker.lm_query_num)

        if attack_result is None:
            str1 = f'\tFail'
            self.log_write(str1)
        else:
            # change ratio
            num_changes = self.cal_change_num(attack_result ,attack_input)
            change_ratio = num_changes / x_len
            # print(f'\tModification Rate: {change_ratio:.2%}', )
            str1 = f'\tModification Rate: {change_ratio:.2%}'
            self.log_write(str1)

            if change_ratio > SUCCESS_THRESHOLD:
                self.long_fail_count += 1

                str1 = f'\tFail: Change Rate too High {change_ratio:.2%}'
                # print(str1)
                self.log_write(str1)
            else:
                self.success_count += 1
                self.change_ratio_list.append(change_ratio)
                self.success_query_num_list.append(attacker.query_num)
                self.success_lm_query_num_list.append(attacker.lm_query_num)
                str1 = '\tSuccess'
                # print(str1)
                self.log_write(str1)
                # visulaize_result(model, attack_input, attack_result)


        str1 = f'\tSuccess rate: {(self.success_count / self.test_count): .2%}'
        str2 = '===================== Result END ======================='
        # print(str1)
        # print(str2)
        self.log_write(str1)
        self.log_write(str2)

        self._flush()


    def summary(self):
        str_sum = '====================== Summary ===========================\n' + \
                  f'Success rate: {self.success_count} / {self.test_count} = {(self.success_count / self.test_count): .2%}\n' + \
                  f'Modification Rate: {np.mean(self.change_ratio_list): .2%}\n' + \
                  f'Query Number: {np.mean(self.query_num_list)}\n' + \
                  f'Success Query Number: {np.mean(self.success_query_num_list)}\n' + \
                  f'LM Query Number: {np.mean(self.lm_query_num_list)}\n' + \
                  f'Success LM Query Number: {np.mean(self.success_lm_query_num_list)}\n' + \
                  f'Long fail number: {self.long_fail_count}, Real success rate {(self.success_count + self.long_fail_count) / self.test_count: .2%}' + \
                  '\n\n'
        # print(str_sum)
        self.log_write(str_sum)
        self._flush()

    def close_file(self):
        if self.log_file is not None:
            self.log_file.close()


class GAIMDBLogger(GALogger):

    def cal_change_num(self, attack_result, attack_input):
        num_changes = np.sum(attack_result != attack_input)
        return num_changes


class GALoggerSNLI(GALogger):

    def cal_change_num(self, attack_result, attack_input):
        h1 = attack_input[1].reshape([-1])
        h2 = attack_result[1].reshape([-1])
        num_changes = np.sum(h1 != h2)
        return num_changes

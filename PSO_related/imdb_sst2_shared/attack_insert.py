from __future__ import division

import copy
import numpy as np
from scipy.special import softmax

from src.pso_reimplements.pso_shared import EPSILON


# import tensorflow as tf

class InsertAttack(object):

    def __init__(self, model, candidate):
        self.candidate = candidate
        self.invoke_dict = {}
        self.model = model

        # ===========================================
        self.query_num = 0
        self.query_num_cached = 0

        self.is_reach_goal = False
        self.best_score = 0

        # local search paras
        self.epsilon = EPSILON

    def do_replace(self, x_cur, pos, new_word):
        x_new = x_cur.copy()
        x_new[pos] = new_word
        return x_new

    def predict_batch(self, sentences):

        return np.array([self.predict(s) for s in sentences])

    def predict(self, sentence):
        self.query_num += 1
        if tuple(sentence) in self.invoke_dict:
            self.query_num_cached += 1
            return self.invoke_dict[tuple(sentence)]
        tem = self.model.predict(np.array([sentence]))[0]
        tem = softmax(tem)
        self.invoke_dict[tuple(sentence)] = tem

        return tem

    def clean_predict(self, sentence):
        if tuple(sentence) in self.invoke_dict:
            self.query_num_cached += 1

        tem = self.model.predict(np.array([sentence]))[0]
        tem = softmax(tem)

        return tem

    # ============================= new algorithm =========================================
    def _is_eq(self, x1, x2):
        for i in range(self.x_len):
            if x1[i] != x2[i]:
                return False
        return True

    def _find_diff_inidces(self, x1, x2):
        diff_indices = set()
        for i in range(self.x_len):
            if x1[i] != x2[i]:
                diff_indices.add(i)

        return diff_indices

    def _greedy_insert(self, x_adv, x_orig, raw_search_indices_set, raw_candidates_by_wid, target):
        search_indices_set = raw_search_indices_set.copy()
        candidates_by_wid = copy.deepcopy(raw_candidates_by_wid)

        changed_indices_set = set()

        # # add orig word to candidates
        # for w_idx in search_indices_set:
        #     candidates_by_wid[w_idx].append(x_orig[w_idx])

        # x_adv = x_adv.copy()
        while True:
            x_new_adv_list = []
            replace_idx_list = []
            scores = []
            pred_list = []

            for replace_idx in search_indices_set:
                # assert len(candidates_by_wid[replace_idx]) > 0
                if replace_idx in changed_indices_set:
                    continue

                for neighbor in candidates_by_wid[replace_idx]:
                    if neighbor == x_adv[replace_idx]:
                        continue
                    x_new = self.do_replace(x_adv, replace_idx, neighbor)
                    x_new_adv_list.append(x_new)
                    replace_idx_list.append(replace_idx)

            if len(x_new_adv_list) == 0:
                break

            for x_new_adv in x_new_adv_list:
                pred = self.predict(x_new_adv)
                pred_list.append(pred)
                score = pred[target]
                scores.append(score)

            cur_best_idx = np.argmax(scores)
            cur_best_x_adv = x_new_adv_list[cur_best_idx]
            cur_best_score = scores[cur_best_idx]
            cur_best_replace_idx = replace_idx_list[cur_best_idx]

            # check goal is acheived
            if np.argmax(pred_list[cur_best_idx]) == target:
                self.is_reach_goal = True
                return cur_best_x_adv

            if cur_best_score - self.best_score > self.epsilon:
                # update best
                self.best_score = cur_best_score
                x_adv = cur_best_x_adv
                changed_indices_set.add(cur_best_replace_idx)
            else:  # no increase
                break

        return x_adv

    # def _check_supplement(self, x_adv, x_orig, neigbhours_list, target):
    #     '''check the supplementary set
    #     '''
    #
    #     supplement_candidates_list = []
    #     for i in range(self.x_len):
    #         supplement_candidates = []
    #         if len(neigbhours_list[i]) > 0:
    #             for n in neigbhours_list[i]:
    #                 if n != x_adv[i]:
    #                     supplement_candidates.append(n)
    #
    #         supplement_candidates_list.append(supplement_candidates)
    #
    #     # check validation of supplementary set
    #     is_set_valid = True
    #     for supplement_candidates in supplement_candidates_list:
    #         if len(supplement_candidates) > 1:
    #             is_set_valid = False
    #             break
    #
    #     # form and check supplement solution x
    #     if is_set_valid:
    #         x_supp = x_orig.copy()
    #         for i in range(self.x_len):
    #             if len(supplement_candidates_list[i]) > 0:
    #                 assert len(supplement_candidates_list[i]) == 1
    #                 x_supp = self.do_replace(x_supp, i, supplement_candidates_list[i][0])
    #
    #         # compare score
    #         pred_supp = self.predict(x_supp)
    #         score_supp = pred_supp[target]
    #         if np.argmax(pred_supp) == target:
    #             self.is_reach_goal = True
    #             return x_supp
    #
    #         if score_supp > self.best_score:
    #             self.best_score = score_supp
    #             x_adv = x_supp
    #
    #     return x_adv

    def local_search(self, x_adv, x_orig, candidates_by_wid, target):

        # ls_max_iter = self.max_iters
        search_indices_set = set()
        for i in range(self.x_len):
            if len(candidates_by_wid[i]) > 0:
                search_indices_set.add(i)

        # local search greedy
        x_cur = x_adv.copy()
        x_cur = self._greedy_insert(x_cur, x_orig, search_indices_set, candidates_by_wid, target)
        if self.is_reach_goal:
            return x_cur

        # # check supplement
        # x_adv = self._check_supplement(x_adv, x_orig, candidates_by_wid, target)
        # if self.is_reach_goal:
        #     return x_adv

        return x_cur

    def _init_attack(self):
        self.query_num = 0
        self.query_num_cached = 0
        self.is_reach_goal = False
        self.invoke_dict = {}

    def attack(self, x_orig, target, pos_tags):
        self._init_attack()

        x_adv = x_orig.copy()
        x_len = np.sum(np.sign(x_orig))
        x_len = int(x_len)
        self.x_len = x_len

        pos_list = ['JJ', 'NN', 'RB', 'VB']

        neigbhours_list = []
        for i in range(x_len):
            if x_adv[i] not in range(1, 50000):
                neigbhours_list.append([])
                continue
            pair = pos_tags[i]
            if pair[1][:2] not in pos_list:
                neigbhours_list.append([])
                continue
            if pair[1][:2] == 'JJ':
                pos = 'adj'
            elif pair[1][:2] == 'NN':
                pos = 'noun'
            elif pair[1][:2] == 'RB':
                pos = 'adv'
            else:
                pos = 'verb'
            if pos in self.candidate[x_adv[i]]:
                neigbhours_list.append([neighbor for neighbor in self.candidate[x_adv[i]][pos]])
            else:
                neigbhours_list.append([])

        neighbours_len = [len(x) for x in neigbhours_list]

        orig_score = self.predict(x_orig)
        self.best_score = orig_score[target]
        print('orig', orig_score[target])

        if np.sum(neighbours_len) == 0:
            return None

        x_adv = self.local_search(x_orig, x_orig, neigbhours_list, target)
        if self.is_reach_goal:
            return x_adv

        return None

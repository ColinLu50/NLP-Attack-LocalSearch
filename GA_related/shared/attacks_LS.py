import numpy as np

from GA_related.shared import glove_utils
from GA_related.shared.paras import EPSILON


class LSAttackSNIL(object):
    def __init__(self, model, dist_mat, inv_vocab, lm, n1=8, n2=4, use_suffix=False):
        self.model = model
        self.dist_mat = dist_mat

        # add LM
        self.use_suffix = use_suffix
        self.lm = lm
        self.inv_vocab = inv_vocab

        # added
        self.best_score = 0
        self.is_reach_goal = False

        self.x1 = None
        self.top_n = n1
        self.top_n2 = n2
        self.query_num = 0
        self.lm_query_num = 0

    def do_replace(self, x_cur, pos, new_word):
        x_new = x_cur.copy()
        x_new[pos] = new_word
        return x_new

    def select_best_replacement(self, replace_idx, x_cur, x_orig, target, neighbours):
        """ Select the most effective replacement to word at pos (pos)
        in (x_cur) between the words in replace_list """
        # check add and swap
        new_x_list = []
        lm_check_words = []
        for w in neighbours:
            if x_cur[replace_idx] != w:
                assert w != 0
                new_x_list.append(self.do_replace(x_cur, replace_idx, w))
                if w in self.inv_vocab:
                    lm_check_words.append(self.inv_vocab[w])
                else:
                    lm_check_words.append('UNK')

        # check deletion
        if x_cur[replace_idx] != x_orig[replace_idx]:
            new_x_list.append(self.do_replace(x_cur, replace_idx, x_orig[replace_idx]))

        valid_num = len(new_x_list)
        if valid_num == 0:
            return None

        cur_x1 = np.tile(self.x1, (valid_num, 1, 1)).reshape(valid_num, -1)
        new_x_preds = self.model.predict([cur_x1, np.array(new_x_list)])
        self.query_num += valid_num

        # Keep only top_n
        new_x_scores = new_x_preds[:, target]

        if np.max(new_x_scores) < self.best_score + EPSILON:
            # no better solution for this replacement
            return None

        # LM
        prefix = ""
        suffix = None
        if replace_idx > 0 and x_cur[replace_idx - 1] in self.inv_vocab:
            prefix = self.inv_vocab[x_cur[replace_idx - 1]]
        #
        # orig_word = self.dataset.inv_dict[x_orig[replace_idx]]
        if self.use_suffix and replace_idx < x_cur.shape[0] - 1:
            if (x_cur[replace_idx + 1] != 0):
                suffix = self.inv_vocab[x_cur[replace_idx + 1]]

        # replace_words_and_orig = [self.dataset.inv_dict[w] if w in self.dataset.inv_dict else 'UNK' for w in
        #                           neighbours[:self.top_n]] + [orig_word]

        replace_words_lm_scores = self.lm.get_words_probs(prefix, lm_check_words, suffix)
        self.lm_query_num += len(lm_check_words)

        # select words
        new_words_lm_scores = np.array(replace_words_lm_scores[:-1])
        # abs_diff_lm_scores = np.abs(new_words_lm_scores - replace_words_lm_scores[-1])
        # rank_replaces_by_lm = np.argsort(abs_diff_lm_scores)
        rank_replaces_by_lm = np.argsort(-new_words_lm_scores)

        filtered_words_idx = rank_replaces_by_lm[self.top_n2:]
        # print(filtered_words_idx)
        new_x_scores[filtered_words_idx] = -10000000

        if np.max(new_x_scores) > self.best_score + EPSILON:
            cur_best = np.argsort(new_x_scores)[-1]
            return new_x_list[cur_best], new_x_scores[cur_best], new_x_preds[cur_best]

        return None

    def _greedy_all(self, x_adv, x_orig, raw_search_indices_set, raw_candidates_by_wid, target):
        search_indices_set = raw_search_indices_set

        while True:
            x_new_list = []
            score_new_list = []
            pred_new_list = []

            for replace_idx in search_indices_set:
                # assert len(candidates_by_wid[replace_idx]) > 0
                res = self.select_best_replacement(replace_idx, x_adv, x_orig, target,
                                                   raw_candidates_by_wid[replace_idx])
                if res is not None:
                    x_new, score_new, pred_new = res[0], res[1], res[2]
                    x_new_list.append(x_new)
                    score_new_list.append(score_new)
                    pred_new_list.append(pred_new)

            if len(x_new_list) == 0:
                break

            cur_best_idx = np.argsort(score_new_list)[-1]
            cur_best_pred = pred_new_list[cur_best_idx]
            cur_best_score = score_new_list[cur_best_idx]
            cur_best_x = x_new_list[cur_best_idx]

            # check goal is acheived
            if np.argmax(cur_best_pred) == target:
                self.is_reach_goal = True
                return cur_best_x

            if cur_best_score - self.best_score > EPSILON:
                # update best
                self.best_score = cur_best_score
                x_adv = cur_best_x
            else:  # no increase
                break

        return x_adv

    def _check_supplement(self, x_adv, x_orig, neigbhours_list, target):
        '''check the supplementary set
        '''

        supplement_candidates_list = []
        for i in range(self.x_len_all):
            supplement_candidates = []
            if len(neigbhours_list[i]) > 0:
                for n in neigbhours_list[i]:
                    if n != x_adv[i]:
                        supplement_candidates.append(n)

            supplement_candidates_list.append(supplement_candidates)

        # check validation of supplementary set
        is_set_valid = True
        for supplement_candidates in supplement_candidates_list:
            if len(supplement_candidates) > 1:
                is_set_valid = False
                break

        # form and check supplement solution x
        if is_set_valid:
            x_supp = x_orig.copy()
            for i in range(self.x_len_all):
                if len(supplement_candidates_list[i]) > 0:
                    assert len(supplement_candidates_list[i]) == 1
                    x_supp = self.do_replace(x_supp, i, supplement_candidates_list[i][0])

            # compare score
            pred_supp = self.model.predict([self.x1[np.newaxis, :], x_supp[np.newaxis, :]])[0]
            self.query_num += 1
            score_supp = pred_supp[target]
            if np.argmax(pred_supp) == target:
                self.is_reach_goal = True
                return x_supp

            if score_supp > self.best_score:
                self.best_score = score_supp
                x_adv = x_supp

        return x_adv

    def _reset_state(self):
        self.query_num = 0
        self.lm_query_num = 0
        self.is_reach_goal = False

    def attack(self, x_orig, target):
        self._reset_state()

        x1_adv = x_orig[0].copy().ravel()
        x2_adv = x_orig[1].copy().ravel()
        x1_orig = x_orig[0].ravel()
        x2_orig = x_orig[1].ravel()
        x1_len = np.sum(np.sign(x1_adv))
        x2_len = np.sum(np.sign(x2_adv))

        # add
        self.x1 = x1_orig
        self.x_len_all = len(x2_adv)
        self.best_score = self.model.predict(x_orig)[0, target]

        tmp = [glove_utils.pick_most_similar_words(x2_adv[i], self.dist_mat, self.top_n, 0.5)
               if x2_adv[i] != 0 else ([], []) for i in range(len(x2_adv))]
        neighbours_list = [x[0] for x in tmp]
        # neighbours_dist = [x[1] for x in tmp]

        search_indices_set = set()
        WORD_THRESHOLD = 17
        for i in range(self.x_len_all):
            if x2_orig[i] >= WORD_THRESHOLD and len(neighbours_list[i]) > 0:
                # >= 27 to prevent replacement of words like 'the', 'a', 'of', etc.
                search_indices_set.add(i)

        # local search
        x_adv = x2_orig
        x_adv = self._greedy_all(x_adv, x2_orig, search_indices_set, neighbours_list, target)
        if self.is_reach_goal:
            return self.x1, x_adv

        # check supplement
        x_adv = self._check_supplement(x_adv, x2_orig, neighbours_list, target)
        if self.is_reach_goal:
            return self.x1, x_adv

        return None


class LSAttackIMDB(object):
    def __init__(self, sess, model, batch_model,
                 neighbour_model,
                 dataset, dist_mat,
                 # skip_list,
                 lm,
                 # pop_size=20, max_iters=100,
                 n1=20, n2=5,
                 use_lm=True, use_suffix=False):
        self.dist_mat = dist_mat
        self.dataset = dataset
        self.dict = self.dataset.dict
        self.inv_dict = self.dataset.inv_dict
        # self.skip_list = skip_list
        self.model = model
        self.batch_model = batch_model
        self.neighbour_model = neighbour_model
        self.sess = sess

        self.lm = lm
        self.top_n = n1  # similar words
        self.top_n2 = n2
        self.use_lm = use_lm
        self.use_suffix = use_suffix

        # add
        self.is_reach_goal = False
        self.best_score = 0
        self.x_len = 0

        self.query_num = 0
        self.lm_query_num = 0

    def _reset_state(self):
        self.query_num = 0
        self.lm_query_num = 0
        self.is_reach_goal = False

    def do_replace(self, x_cur, pos, new_word):
        x_new = x_cur.copy()
        x_new[pos] = new_word
        return x_new

    def select_best_replacement(self, replace_idx, x_cur, x_orig, target, neighbours):
        """ Select the most effective replacement to word at replace idx (replace_idx)
        in (x_cur) between the words in replace_list """

        # check add and swap
        new_x_list = []
        lm_check_words = []
        for w in neighbours:
            if x_cur[replace_idx] != w:
                assert w != 0
                new_x_list.append(self.do_replace(x_cur, replace_idx, w))
                if w in self.dataset.inv_dict:
                    lm_check_words.append(self.dataset.inv_dict[w])
                else:
                    lm_check_words.append('UNK')

        # check deletion
        if x_cur[replace_idx] != x_orig[replace_idx]:
            new_x_list.append(self.do_replace(x_cur, replace_idx, x_orig[replace_idx]))

        valid_num = len(new_x_list)

        # pad for batch predict (faster)
        for i in range(self.top_n - valid_num):
            new_x_list.append(x_cur)

        new_x_preds = self.neighbour_model.predict(self.sess, np.array(new_x_list))
        self.query_num += valid_num

        new_x_scores = new_x_preds[:valid_num, target]
        new_x_preds = new_x_preds[:valid_num, :]
        if np.max(new_x_scores) < self.best_score + EPSILON:
            # no better solution for this replacement
            return None

        if self.use_lm:
            prefix = ""
            suffix = None
            if replace_idx > 0:
                prefix = self.dataset.inv_dict[x_cur[replace_idx - 1]]
            #
            # orig_word = self.dataset.inv_dict[x_orig[replace_idx]]
            if self.use_suffix and replace_idx < x_cur.shape[0] - 1:
                if (x_cur[replace_idx + 1] != 0):
                    suffix = self.dataset.inv_dict[x_cur[replace_idx + 1]]

            # replace_words_and_orig = [self.dataset.inv_dict[w] if w in self.dataset.inv_dict else 'UNK' for w in
            #                           neighbours[:self.top_n]] + [orig_word]

            replace_words_lm_scores = self.lm.get_words_probs(prefix, lm_check_words, suffix)
            self.lm_query_num += len(lm_check_words)

            # select words
            new_words_lm_scores = np.array(replace_words_lm_scores[:-1])
            # abs_diff_lm_scores = np.abs(new_words_lm_scores - replace_words_lm_scores[-1])
            # rank_replaces_by_lm = np.argsort(abs_diff_lm_scores)
            rank_replaces_by_lm = np.argsort(-new_words_lm_scores)

            filtered_words_idx = rank_replaces_by_lm[self.top_n2:]
            # print(filtered_words_idx)
            new_x_scores[filtered_words_idx] = -10000000

        if np.max(new_x_scores) > self.best_score + EPSILON:
            cur_best = np.argsort(new_x_scores)[-1]
            return new_x_list[cur_best], new_x_scores[cur_best], new_x_preds[cur_best, :]

        return None

    def _greedy_all(self, x_adv, x_orig, raw_search_indices_set, raw_candidates_by_wid, target):
        search_indices_set = raw_search_indices_set.copy()

        # x_adv = x_adv.copy()
        while True:

            x_new_list = []
            score_new_list = []
            pred_new_list = []

            for replace_idx in search_indices_set:
                # assert len(candidates_by_wid[replace_idx]) > 0
                res = self.select_best_replacement(replace_idx, x_adv, x_orig, target,
                                                   raw_candidates_by_wid[replace_idx])
                if res is not None:
                    x_new, score_new, pred_new = res[0], res[1], res[2]
                    x_new_list.append(x_new)
                    score_new_list.append(score_new)
                    pred_new_list.append(pred_new)

            if len(x_new_list) == 0:
                break

            cur_best_idx = np.argsort(score_new_list)[-1]
            cur_best_pred = pred_new_list[cur_best_idx]
            cur_best_score = score_new_list[cur_best_idx]
            cur_best_x = x_new_list[cur_best_idx]

            # check goal is acheived
            if np.argmax(cur_best_pred) == target:
                self.is_reach_goal = True
                return cur_best_x

            if cur_best_score - self.best_score > EPSILON:
                # update best
                self.best_score = cur_best_score
                x_adv = cur_best_x
            else:  # no increase
                break

        return x_adv

    def _check_supplement(self, x_adv, x_orig, neigbhours_list, target):
        '''check the supplementary set
        '''

        supplement_candidates_list = []
        for i in range(self.x_len):
            supplement_candidates = []
            if len(neigbhours_list[i]) > 0:
                for n in neigbhours_list[i]:
                    if n != x_adv[i]:
                        supplement_candidates.append(n)

            supplement_candidates_list.append(supplement_candidates)

        # check validation of supplementary set
        is_set_valid = True
        for supplement_candidates in supplement_candidates_list:
            if len(supplement_candidates) > 1:
                is_set_valid = False
                break

        # form and check supplement solution x
        if is_set_valid:
            x_supp = x_orig.copy()
            for i in range(self.x_len):
                if len(supplement_candidates_list[i]) > 0:
                    assert len(supplement_candidates_list[i]) == 1
                    x_supp = self.do_replace(x_supp, i, supplement_candidates_list[i][0])

            # compare score
            pred_supp = self.predict(x_supp)
            score_supp = pred_supp[target]
            if np.argmax(pred_supp) == target:
                self.is_reach_goal = True
                return x_supp

            if score_supp > self.best_score:
                self.best_score = score_supp
                x_adv = x_supp

        return x_adv

    def attack(self, x_orig, target):
        self._reset_state()

        x_adv = x_orig.copy()
        x_len = int(np.sum(np.sign(x_orig)))
        self.x_len = x_len

        self.best_score = self.model.predict(self.sess, x_orig[np.newaxis, :])[0, target]

        tmp = [glove_utils.pick_most_similar_words(
            x_orig[i], self.dist_mat, self.top_n, 0.5) for i in range(x_len)]
        neighbours_list = [x[0] for x in tmp]  # neighbors_list[i] is a nparray
        # neighbours_dist = [x[1] for x in tmp]

        search_indices_set = set()
        for i in range(x_len):
            if x_orig[i] >= 27 and len(neighbours_list[i]) > 0:
                # >= 27 to prevent replacement of words like 'the', 'a', 'of', etc.
                search_indices_set.add(i)

        # local search
        x_adv = self._greedy_all(x_adv, x_orig, search_indices_set, neighbours_list, target)
        if self.is_reach_goal:
            return x_adv

        # check supplement
        x_adv = self._check_supplement(x_adv, x_orig, neighbours_list, target)
        if self.is_reach_goal:
            return x_adv

        return None

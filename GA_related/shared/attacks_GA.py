from time import time
import numpy as np


from GA_related.shared import glove_utils


class EntailmentAttack(object):
    def __init__(self, model, dist_mat, inv_vocab, lm, pop_size=4, max_iters=10, n1=8, n2=4, use_suffix=False):
        self.model = model
        self.dist_mat = dist_mat
        self.n1 = n1
        self.n2 = n2

        self.max_iters = max_iters
        self.pop_size = pop_size
        self.temp = 1.0

        # add LM
        self.use_suffix = use_suffix
        self.lm = lm
        self.inv_vocab = inv_vocab

        # added
        # @TODO: record and reset query number
        self.x1 = None
        self.top_n = n1
        self.top_n2 = n2
        self.query_num = 0
        self.lm_query_num = 0

    def do_replace(self, x_cur, pos, new_word):
        x_new = x_cur.copy()
        x_new[pos] = new_word
        return x_new


    def select_best_replacement(self, pos, x_cur, x_orig, target, replace_list):
        """ Select the most effective replacement to word at pos (pos)
        in (x_cur) between the words in replace_list """
        new_x_list = []
        valid_num = 0
        for w in replace_list:
            if x_orig[pos] != w and w != 0:
                new_x_list.append(self.do_replace(x_cur, pos, w))
                valid_num += 1

        pred_cur = self.model.predict([self.x1[np.newaxis, :], x_cur[np.newaxis, :]])[0]
        self.query_num += 1

        if valid_num == 0:
            return x_cur, pred_cur

        cur_x1 = np.tile(self.x1, (valid_num, 1, 1)).reshape(valid_num, -1)
        new_preds = self.model.predict([cur_x1, np.array(new_x_list)])
        self.query_num += valid_num

        # Keep only top_n
        new_x_scores = new_preds[:, target]
        score_cur = pred_cur[target]
        new_x_scores = new_x_scores - score_cur
        # Eliminate not that close words
        new_x_scores[self.top_n:] = -10000000

        # LM
        prefix = ""
        suffix = None
        if pos > 0 and x_cur[pos - 1] > 0:
            prefix = self.inv_vocab[x_cur[pos - 1]]

        orig_word = self.inv_vocab[x_orig[pos]]
        if self.use_suffix and pos < x_cur.shape[0] - 1:
            if (x_cur[pos + 1] != 0):
                suffix = self.inv_vocab[x_cur[pos + 1]]
        # print('** ', orig_word)
        # TODO: should remove orig_word or self.top_n2 = K + 1
        replace_words_and_orig = [self.inv_vocab[w] if w in self.inv_vocab else 'UNK' for w in
                                  replace_list[:self.top_n]] + [orig_word]
        # replace_words_and_orig = [self.dataset.inv_dict[w] if w in self.dataset.inv_dict else 'UNK' for w in
        #                           replace_list[:self.top_n]]
        # print(replace_words_and_orig)
        replace_words_lm_scores = self.lm.get_words_probs(prefix, replace_words_and_orig, suffix)
        self.lm_query_num += len(replace_words_and_orig)
        # print(replace_words_lm_scores)
        # for i in range(len(replace_words_and_orig)):
        #    print(replace_words_and_orig[i], ' -- ', replace_words_lm_scores[i])

        # select words
        new_words_lm_scores = np.array(replace_words_lm_scores[:-1])
        # abs_diff_lm_scores = np.abs(new_words_lm_scores - replace_words_lm_scores[-1])
        # rank_replaces_by_lm = np.argsort(abs_diff_lm_scores)
        rank_replaces_by_lm = np.argsort(-new_words_lm_scores)

        filtered_words_idx = rank_replaces_by_lm[self.top_n2:]
        # print(filtered_words_idx)
        new_x_scores[filtered_words_idx] = -10000000

        if (np.max(new_x_scores) > 0):
            cur_best_idx = np.argsort(new_x_scores)[-1]
            return new_x_list[cur_best_idx], new_preds[cur_best_idx]
        return x_cur, pred_cur

    def perturb(self, x_cur, x_orig, neighbours_list, w_select_probs, target):
        # rand_idx = np.random.choice(
        #     w_select_probs.shape[0], 1, p=w_select_probs)[0]
        # new_w = np.random.choice(neighbours_list[rand_idx])
        # return self.do_replace(x_cur, rand_idx, new_w)

        # Pick a word that is not modified and is not UNK
        x_len = w_select_probs.shape[0]
        rand_idx = np.random.choice(x_len, 1, p=w_select_probs)[0]
        while x_cur[rand_idx] != x_orig[rand_idx] and np.sum(x_orig != x_cur) < np.sum(np.sign(w_select_probs)):
            rand_idx = np.random.choice(x_len, 1, p=w_select_probs)[0]

        replace_list = neighbours_list[rand_idx]
        return self.select_best_replacement(rand_idx, x_cur, x_orig, target, replace_list)

    def generate_population(self, x_orig, neighbours_list, w_select_probs, target):
        new_x_list = []
        new_pred_list = []
        for _ in range(self.pop_size):
            x_new, pred_new = self.perturb(x_orig, x_orig, neighbours_list, w_select_probs, target)
            new_x_list.append(x_new)
            new_pred_list.append(pred_new)
        return new_x_list, new_pred_list

    def crossover(self, x1, x2):
        x_new = x1.copy()
        for i in range(len(x1)):
            if np.random.uniform() < 0.5:
                x_new[i] = x2[i]
        return x_new

    def _reset_state(self):
        self.query_num = 0
        self.lm_query_num = 0

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

        tmp = [glove_utils.pick_most_similar_words(x2_adv[i], self.dist_mat, 50, 0.5) if x2_adv[i] != 0 else ([], [])
               for i in range(len(x2_adv))]
        neighbours_list = [x[0] for x in tmp]
        neighbours_dist = [x[1] for x in tmp]
        neigbhours_len = [len(x) for x in neighbours_list]
        w_select_probs = neigbhours_len / np.sum(neigbhours_len)

        # BUG: w_select_probs have NaN
        if np.sum(neigbhours_len) == 0:
            return None

        tmp = [glove_utils.pick_most_similar_words(x2_adv[i], self.dist_mat, self.n1, 0.5)
               if x2_adv[i] != 0 else ([], []) for i in range(len(x2_adv))]
        neighbours_list = [x[0] for x in tmp]
        # neighbours_dist = [x[1] for x in tmp]

        new_x_list, new_pred_list = self.generate_population(x2_adv, neighbours_list, w_select_probs, target)
        pop = np.array(new_x_list)
        pop_preds = np.array(new_pred_list)
        pop = pop.reshape(self.pop_size, -1)

        for iter_idx in range(self.max_iters):

            pop_scores = pop_preds[:, target]
            top_attack = np.argsort(pop_scores)[-1]
            if np.argmax(pop_preds[top_attack, :]) == target:
                return x1_orig, pop[top_attack]
            print('\t', iter_idx, ' : ', np.max(pop_scores))
            logits = np.exp(pop_scores / self.temp)
            pop_select_probs = logits / np.sum(logits)

            elite = [pop[top_attack]]
            elite_pred = [pop_preds[top_attack]]

            parent1_idx = np.random.choice(
                self.pop_size, size=self.pop_size - 1, p=pop_select_probs)
            parent2_idx = np.random.choice(
                self.pop_size, size=self.pop_size - 1, p=pop_select_probs)

            childs = [self.crossover(pop[parent1_idx[i]], pop[parent2_idx[i]])
                      for i in range(self.pop_size - 1)]

            new_x_list = []
            new_pred_list = []
            for x in childs:
                x_new, pred_new = self.perturb(x, x2_orig, neighbours_list, w_select_probs, target)
                new_x_list.append(x_new)
                new_pred_list.append(pred_new)

            pop = elite + new_x_list
            pop = np.array(pop)
            pop_preds_list = elite_pred + new_pred_list
            pop_preds = np.array(pop_preds_list)

        return None


class GeneticAtack(object):
    def __init__(self, sess, model, batch_model,
                 neighbour_model,
                 dataset, dist_mat,
                 skip_list,
                 lm,
                 pop_size=20, max_iters=100,
                 n1=20, n2=5,
                 use_lm=True, use_suffix=False):
        self.dist_mat = dist_mat
        self.dataset = dataset
        self.dict = self.dataset.dict
        self.inv_dict = self.dataset.inv_dict
        self.skip_list = skip_list
        self.model = model
        self.batch_model = batch_model
        self.neighbour_model = neighbour_model
        self.sess = sess
        self.max_iters = max_iters
        self.pop_size = pop_size
        self.lm = lm
        self.top_n = n1  # similar words
        self.top_n2 = n2
        self.use_lm = use_lm
        self.use_suffix = use_suffix
        self.temp = 0.3

        # add
        self.query_num = 0
        self.lm_query_num = 0

    def _reset_state(self):
        self.query_num = 0
        self.lm_query_num = 0

    def do_replace(self, x_cur, pos, new_word):
        x_new = x_cur.copy()
        x_new[pos] = new_word
        return x_new

    def select_best_replacement(self, pos, x_cur, x_orig, target, replace_list):
        """ Select the most effective replacement to word at pos (pos)
        in (x_cur) between the words in replace_list """
        new_x_list = []
        valid_num = 0
        for w in replace_list:
            if x_orig[pos] != w and w != 0:
                new_x_list.append(self.do_replace(x_cur, pos, w))
                valid_num += 1
            else:
                new_x_list.append(x_cur)

        # new_x_list = [self.do_replace(
        #     x_cur, pos, w) if x_orig[pos] != w and w != 0 else x_cur for w in replace_list]
        new_x_preds = self.neighbour_model.predict(
            self.sess, np.array(new_x_list))
        self.query_num += valid_num

        # Keep only top_n
        # replace_list = replace_list[:self.top_n]
        # new_x_list = new_x_list[:self.top_n]
        # new_x_preds = new_x_preds[:self.top_n,:]
        new_x_scores = new_x_preds[:, target]
        orig_score = self.model.predict(
            self.sess, x_cur[np.newaxis, :])[0, target]
        self.query_num += 1
        new_x_scores = new_x_scores - orig_score
        # Eliminate not that close words
        new_x_scores[self.top_n:] = -10000000

        if self.use_lm:
            prefix = ""
            suffix = None
            if pos > 0:
                prefix = self.dataset.inv_dict[x_cur[pos - 1]]
            #
            orig_word = self.dataset.inv_dict[x_orig[pos]]
            if self.use_suffix and pos < x_cur.shape[0] - 1:
                if (x_cur[pos + 1] != 0):
                    suffix = self.dataset.inv_dict[x_cur[pos + 1]]
            # print('** ', orig_word)
            # TODO: should remove orig_word or self.top_n2 = K + 1
            replace_words_and_orig = [self.dataset.inv_dict[w] if w in self.dataset.inv_dict else 'UNK' for w in
                                      replace_list[:self.top_n]] + [orig_word]
            # replace_words_and_orig = [self.dataset.inv_dict[w] if w in self.dataset.inv_dict else 'UNK' for w in
            #                           replace_list[:self.top_n]]
            # print(replace_words_and_orig)
            replace_words_lm_scores = self.lm.get_words_probs(prefix, replace_words_and_orig, suffix)
            self.lm_query_num += len(replace_words_and_orig)
            # print(replace_words_lm_scores)
            # for i in range(len(replace_words_and_orig)):
            #    print(replace_words_and_orig[i], ' -- ', replace_words_lm_scores[i])

            # select words
            new_words_lm_scores = np.array(replace_words_lm_scores[:-1])
            # abs_diff_lm_scores = np.abs(new_words_lm_scores - replace_words_lm_scores[-1])
            # rank_replaces_by_lm = np.argsort(abs_diff_lm_scores)
            rank_replaces_by_lm = np.argsort(-new_words_lm_scores)

            filtered_words_idx = rank_replaces_by_lm[self.top_n2:]
            # print(filtered_words_idx)
            new_x_scores[filtered_words_idx] = -10000000

        if (np.max(new_x_scores) > 0):
            return new_x_list[np.argsort(new_x_scores)[-1]]
        return x_cur

    def perturb(self, x_cur, x_orig, neigbhours, neighbours_dist, w_select_probs, target):

        # Pick a word that is not modified and is not UNK
        x_len = w_select_probs.shape[0]
        # to_modify = [idx  for idx in range(x_len) if (x_cur[idx] == x_orig[idx] and self.inv_dict[x_cur[idx]] != 'UNK' and
        #                                             self.dist_mat[x_cur[idx]][x_cur[idx]] != 100000) and
        #                     x_cur[idx] not in self.skip_list
        #            ]
        rand_idx = np.random.choice(x_len, 1, p=w_select_probs)[0]
        while x_cur[rand_idx] != x_orig[rand_idx] and np.sum(x_orig != x_cur) < np.sum(np.sign(w_select_probs)):
            # The conition above has a quick hack to prevent getting stuck in infinite loop while processing too short examples
            # and all words `excluding articles` have been already replaced and still no-successful attack found.
            # a more elegent way to handle this could be done in attack to abort early based on the status of all population members
            # or to improve select_best_replacement by making it schocastic.
            rand_idx = np.random.choice(x_len, 1, p=w_select_probs)[0]
        # print(f'while loop time: {tmp_while_t2 - tmp_while_t1: .5f}s')

        # src_word = x_cur[rand_idx]
        # replace_list,_ =  glove_utils.pick_most_similar_words(src_word, self.dist_mat, self.top_n, 0.5)
        replace_list = neigbhours[rand_idx]
        if len(replace_list) < self.top_n:
            replace_list = np.concatenate(
                (replace_list, np.zeros(self.top_n - replace_list.shape[0])))
        return self.select_best_replacement(rand_idx, x_cur, x_orig, target, replace_list)

    def generate_population(self, x_orig, neigbhours_list, neighbours_dist, w_select_probs, target, pop_size):
        return [self.perturb(x_orig, x_orig, neigbhours_list, neighbours_dist, w_select_probs, target) for _ in
                range(pop_size)]

    def crossover(self, x1, x2):
        x_new = x1.copy()
        for i in range(len(x1)):
            if np.random.uniform() < 0.5:
                x_new[i] = x2[i]
        return x_new

    def attack(self, x_orig, target, max_change=0.4):
        self._reset_state()

        x_adv = x_orig.copy()
        x_len = np.sum(np.sign(x_orig))
        # Neigbhours for every word.
        tmp = [glove_utils.pick_most_similar_words(
            x_orig[i], self.dist_mat, 50, 0.5) for i in range(x_len)]
        neigbhours_list = [x[0] for x in tmp]
        neighbours_dist = [x[1] for x in tmp]
        neighbours_len = [len(x) for x in neigbhours_list]
        for i in range(x_len):
            if (x_adv[i] < 27):
                # To prevent replacement of words like 'the', 'a', 'of', etc.
                neighbours_len[i] = 0
        w_select_probs = neighbours_len / np.sum(neighbours_len)
        tmp = [glove_utils.pick_most_similar_words(
            x_orig[i], self.dist_mat, self.top_n, 0.5) for i in range(x_len)]
        neigbhours_list = [x[0] for x in tmp]
        neighbours_dist = [x[1] for x in tmp]

        # # test
        # tmp_all_neighbour_number = 0
        # tmp_have_candidates_number = 0
        # for neighbours in neigbhours_list:
        #     tmp_all_neighbour_number += len(neighbours)
        #     if len(neighbours) > 0:
        #         tmp_have_candidates_number += 1
        # print(f'total have {tmp_all_neighbour_number} candidates, '
        #       f'{tmp_have_candidates_number} words have candidates')
        # tmp_t1 = time()
        pop = self.generate_population(
            x_orig, neigbhours_list, neighbours_dist, w_select_probs, target, self.pop_size)
        # print(f'opt init time {time() - tmp_t1: .5f}s')

        for i in range(self.max_iters):
            # print(i)
            pop_preds = self.batch_model.predict(self.sess, np.array(pop))

            pop_scores = pop_preds[:, target]
            print('\t\t', i, ' -- ', np.max(pop_scores))
            pop_ranks = np.argsort(pop_scores)[::-1]
            top_attack = pop_ranks[0]

            logits = np.exp(pop_scores / self.temp)
            select_probs = logits / np.sum(logits)

            if np.argmax(pop_preds[top_attack, :]) == target:
                return pop[top_attack]
            elite = [pop[top_attack]]  # elite
            # print(select_probs.shape)
            parent1_idx = np.random.choice(
                self.pop_size, size=self.pop_size - 1, p=select_probs)
            parent2_idx = np.random.choice(
                self.pop_size, size=self.pop_size - 1, p=select_probs)

            childs = [self.crossover(pop[parent1_idx[i]],
                                     pop[parent2_idx[i]])
                      for i in range(self.pop_size - 1)]

            # tmp_t2 = time()
            childs = [self.perturb(
                x, x_orig, neigbhours_list, neighbours_dist, w_select_probs, target) for x in childs]
            pop = elite + childs

            # print(f'opt perturb time {time() - tmp_t2: .5f}s')

        return None

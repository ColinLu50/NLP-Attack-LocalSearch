# -*- coding: utf-8 -*-

from abc import abstractmethod
import numpy as np
import sacrebleu
from fairseq.models.transformer import TransformerModel
import nltk, torch
from sacremoses import MosesTokenizer, MosesDetokenizer

from SGS_related.LS.local_search_base import LocalSearchBase
from SGS_related.LS.ls_shared import EPSILON



class LocalSearchNMT(LocalSearchBase):
    @abstractmethod
    def __init__(self):
        pass

    def attack_one(self, source, reference, constrain_pos=True):
        # Return Format (raw, translation, is attack success, query number, modif_rate)

        orig_tokenized = MosesTokenizer(lang='en').tokenize(source)
        # skip too long or too short question
        if len(orig_tokenized) < 10 or len(orig_tokenized) > 100:
            return source, reference, None, None, None

        # generate candidates
        pos_tagged = [(tagged[0], '.') if '&' in tagged[0] else tagged for tagged in
                      nltk.pos_tag(orig_tokenized, tagset='universal')]

        token_inflections = self.get_inflections(orig_tokenized, pos_tagged, constrain_pos)

        # get original bleu
        original_bleu, orig_predicted = self.get_bleu(source, reference)

        # skip examples already have glue == 0
        if original_bleu == 0:
            return source, reference, None, None, None

        _perturbed_tokenized, _bleu, _predicted, _num_queries = self.local_search_nmt(token_inflections,
                                                                           orig_tokenized,
                                                                           source,
                                                                           original_bleu,
                                                                           reference)

        if _bleu == original_bleu:
            _predicted = orig_predicted

        # attack success
        is_attack_success = False
        if _bleu == 0:
            is_attack_success = True

        modif_rate = self.get_modif_rate(orig_tokenized, _perturbed_tokenized)
        attack_text = MosesDetokenizer(lang='en').detokenize(_perturbed_tokenized)
        return attack_text, _predicted, is_attack_success, _num_queries, modif_rate



    def local_search_nmt(self, token_inflections, orig_tokenized, original,
                   original_bleu, reference, backward=False):
        perturbed_tokenized = orig_tokenized.copy()

        best_bleu = original_bleu
        num_queries = 0
        best_predicted = ''

        detokenizer = MosesDetokenizer(lang='en')

        while True:
            new_tokenized_list = []
            new_bleu_list = []
            new_predicted_list = []

            for position, candidates in token_inflections: # list of pairs (position, candidates) candidates: list of token
                # add or swap
                for infl in candidates:

                    if perturbed_tokenized[position] == infl:
                        continue

                    # do replace
                    new_tokenized = perturbed_tokenized.copy()
                    new_tokenized[position] = infl
                    # form text and eval
                    new_text = detokenizer.detokenize(new_tokenized)
                    new_bleu, new_predicted = self.get_bleu(new_text, reference)
                    num_queries += 1

                    # record
                    new_tokenized_list.append(new_tokenized)
                    new_bleu_list.append(new_bleu)
                    new_predicted_list.append(new_predicted)

                # remove
                if perturbed_tokenized[position] != orig_tokenized[position]:
                    # do replace
                    new_tokenized = perturbed_tokenized.copy()
                    new_tokenized[position] = orig_tokenized[position]

                    # form text and eval
                    new_text = detokenizer.detokenize(new_tokenized)
                    new_bleu, new_predicted = self.get_bleu(new_text, reference)
                    num_queries += 1

                    # record
                    new_tokenized_list.append(new_tokenized)
                    new_bleu_list.append(new_bleu)
                    new_predicted_list.append(new_predicted)

            if len(new_bleu_list) == 0: # no improve
                break

            cur_best_idx = np.argsort(new_bleu_list)[0]
            cur_best_bleu = new_bleu_list[cur_best_idx]
            cur_best_predicted = new_predicted_list[cur_best_idx]
            cur_best_tokenized = new_tokenized_list[cur_best_idx]

            # check stop criteria
            if cur_best_bleu == 0:
                perturbed_tokenized = cur_best_tokenized
                best_bleu = cur_best_bleu
                best_predicted = cur_best_predicted
                break

            if cur_best_bleu < best_bleu - EPSILON:
                perturbed_tokenized = cur_best_tokenized
                best_bleu = cur_best_bleu
                best_predicted = cur_best_predicted
            else:
                break

        # =============== check supplement set ======================
        # form supplement set
        supplement_inflections_by_position = {position: [] for position, _ in token_inflections}
        for position, candidates in token_inflections:
            for infl in candidates:
                if perturbed_tokenized[position] != infl:
                    supplement_inflections_by_position[position].append(infl)

        is_sup_valid = True
        valid_positions = []
        for position, _ in token_inflections:
            if len(supplement_inflections_by_position[position]) > 1:
                is_sup_valid = False
                break
            if len(supplement_inflections_by_position[position]) == 1:
                valid_positions.append(position)

        if len(valid_positions) == 0:
            is_sup_valid =False

        if is_sup_valid:
            print('check supplement')
            supplement_tokenized = perturbed_tokenized.copy()
            for position in valid_positions:
                supplement_tokenized[position] = supplement_inflections_by_position[position][0]

            # form text and eval
            supp_text = detokenizer.detokenize(supplement_tokenized)
            supp_bleu, supp_predicted = self.get_bleu(supp_text, reference)
            num_queries += 1

            if supp_bleu < best_bleu:
                best_bleu = supp_bleu
                best_predicted = supp_predicted
                perturbed_tokenized = supplement_tokenized

        return perturbed_tokenized, best_bleu, best_predicted, num_queries

    def get_bleu(self, source, reference, beam=5):
        predicted = self.model.translate(source, beam)
        return sacrebleu.sentence_bleu(predicted, reference).score, predicted

    @staticmethod
    def get_modif_rate(orig_tokenized, permuted_tokenized):
        modif_num = 0
        len_ = len(orig_tokenized)

        for token_idx in range(len_):
            if orig_tokenized[token_idx] != permuted_tokenized[token_idx]:
                modif_num += 1

        return modif_num / len_


'''
Implements model-specific details.
'''


class LSFairseqTransformerNMT(LocalSearchNMT):
    def __init__(self, model_dir, model_file, tokenizer='moses', bpe='subword_nmt', use_cuda=True):
        self.model = TransformerModel.from_pretrained(model_dir, model_file, tokenizer=tokenizer, bpe=bpe)
        if use_cuda and torch.cuda.is_available():
            self.model.cuda()


'''
class MorpheusBiteFairseqTransformerNMT(MorpheusFairseqTransformerNMT):
    def __init__(self, model_dir, model_file, tokenizer='moses', bpe='subword_nmt', use_cuda=True):
        super().__init__(model_dir, model_file, tokenizer, bpe, use_cuda)
        self.bite = BITETokenizer()

    def get_bleu(self, source, reference, beam=5):
        bite_source = ' '.join(self.bite.tokenize(source))
        return super().get_bleu(bite_source, reference, beam)
'''

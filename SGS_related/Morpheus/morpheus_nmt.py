# -*- coding: utf-8 -*-
import sys

sys.path.append("..")
sys.path.append(".")

from abc import abstractmethod
import sacrebleu
from fairseq.models.transformer import TransformerModel
import nltk, torch
from sacremoses import MosesTokenizer, MosesDetokenizer

from SGS_related.Morpheus.morpheus_base import MorpheusBase




class MorpheusNMT(MorpheusBase):
    @abstractmethod
    def __init__(self):
        pass

    def morph(self, source, reference, constrain_pos=True):
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


        forward_perturbed, forward_bleu, forward_predicted, num_queries_forward = self.search_nmt(token_inflections,
                                                                                                  orig_tokenized,
                                                                                                  source,
                                                                                                  original_bleu,
                                                                                                  reference)


        if forward_bleu == original_bleu:
            forward_predicted = orig_predicted

        # attack success
        if forward_bleu == 0:
            modif_rate = self.get_modif_rate(orig_tokenized, forward_perturbed)
            attack_text = MosesDetokenizer(lang='en').detokenize(forward_perturbed)
            return attack_text, forward_predicted, True, num_queries_forward + 1, modif_rate

        backward_perturbed, backward_bleu, backward_predicted, num_queries_backward = self.search_nmt(token_inflections,
                                                                                                      orig_tokenized,
                                                                                                      source,
                                                                                                      original_bleu,
                                                                                                      reference,
                                                                                                      backward=True)

        if backward_bleu == original_bleu:
            backward_predicted = orig_predicted
        num_queries = 1 + num_queries_forward + num_queries_backward
        if forward_bleu < backward_bleu:
            is_attack_success = False
            if forward_bleu == 0:
                is_attack_success = True
            modif_rate = self.get_modif_rate(orig_tokenized, forward_perturbed)
            attack_text = MosesDetokenizer(lang='en').detokenize(forward_perturbed)
            return attack_text, forward_predicted, is_attack_success, num_queries, modif_rate
        else:
            is_attack_success = False
            if backward_bleu == 0:
                is_attack_success = True
            modif_rate = self.get_modif_rate(orig_tokenized, backward_perturbed)
            attack_text = MosesDetokenizer(lang='en').detokenize(backward_perturbed)
            return attack_text, backward_predicted, is_attack_success, num_queries, modif_rate

    def search_nmt(self, token_inflections, orig_tokenized, original,
                   original_bleu, reference, backward=False):
        perturbed_tokenized = orig_tokenized.copy()

        max_bleu = original_bleu
        num_queries = 0
        max_predicted = ''

        if backward:
            token_inflections = reversed(token_inflections)

        detokenizer = MosesDetokenizer(lang='en')

        for curr_token in token_inflections:
            max_infl = orig_tokenized[curr_token[0]]
            for infl in curr_token[1]:
                perturbed_tokenized[curr_token[0]] = infl
                perturbed = detokenizer.detokenize(perturbed_tokenized)
                curr_bleu, predicted = self.get_bleu(perturbed, reference)
                num_queries += 1
                if curr_bleu < max_bleu: # the smaller, the better
                    max_bleu = curr_bleu
                    max_infl = infl
                    max_predicted = predicted
            perturbed_tokenized[curr_token[0]] = max_infl
        return perturbed_tokenized, max_bleu, max_predicted, num_queries

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


class MorpheusFairseqTransformerNMT(MorpheusNMT):
    def __init__(self, model_dir, model_file, tokenizer='moses', bpe='subword_nmt', use_cuda=True):
        self.model = TransformerModel.from_pretrained(model_dir, model_file, tokenizer=tokenizer, bpe=bpe)
        if use_cuda and torch.cuda.is_available():
            self.model.cuda()



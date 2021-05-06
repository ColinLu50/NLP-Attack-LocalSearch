# sys.path.append("..")
# sys.path.append(".")
# from bite_wordpiece import BiteWordpieceTokenizer


from abc import abstractmethod

import nltk
import torch
from allennlp.data.dataset_readers.reading_comprehension.util import char_span_to_token_span
from allennlp.data.tokenizers import WordTokenizer
from allennlp.predictors.predictor import Predictor
from allennlp.tools.squad_eval import metric_max_over_ground_truths
from sacremoses import MosesTokenizer, MosesDetokenizer
from torch.nn import CrossEntropyLoss
# from transformers import BertConfig, BertTokenizer, BertForQuestionAnswering
from transformers import AutoConfig, AutoTokenizer, AutoModelForQuestionAnswering
import numpy as np

from SGS_related.LS.local_search_base import LocalSearchBase
from SGS_related.squad_utils import compute_f1
from SGS_related.LS.ls_shared import EPSILON


class LocalSearchQA(LocalSearchBase):
    @abstractmethod
    def __init__(self):
        pass

    def attack_one(self, question_dict, context, constrain_pos=True):
        original = question_dict['question']

        gold_starts = [ans['answer_start'] for ans in question_dict['answers']]
        gold_texts = [ans['text'] for ans in question_dict['answers']]
        gold_ends = [gold_starts[i] + len(text) for i, text in enumerate(gold_texts)]
        question_dict['gold_char_spans'] = list(zip(gold_starts, gold_ends))
        question_dict['gold_texts'] = gold_texts

        orig_tokenized = MosesTokenizer(lang='en').tokenize(original)

        pos_tagged = [(tagged[0], '.') if '&' in tagged[0] else tagged for tagged in
                      nltk.pos_tag(orig_tokenized, tagset='universal')]

        token_inflections = super(LocalSearchQA, self).get_inflections(orig_tokenized, pos_tagged, constrain_pos)

        original_loss, init_predicted = self.get_loss(original, question_dict, context)

        # skip too long or too short question
        if len(orig_tokenized) < 10 or len(orig_tokenized) > 100:
            return original, 0, None, None

        # skip wrong predict question
        if metric_max_over_ground_truths(compute_f1, init_predicted, question_dict['gold_texts']) == 0:
            return original, 1, None, None

        _perturbed, _loss, _predicted, _num_queries = self.local_search_qa(token_inflections,
                                                                                                 orig_tokenized,
                                                                                                 original_loss,
                                                                                                 question_dict,
                                                                                                 context)
        is_attack_success = False
        if metric_max_over_ground_truths(compute_f1, _predicted, question_dict['gold_texts']) == 0:
            is_attack_success = True

        num_queries = 1 + _num_queries
        modif_rate = self.get_modif_rate(orig_tokenized, _perturbed)
        return MosesDetokenizer(lang='en').detokenize(_perturbed), num_queries, modif_rate, is_attack_success


    def local_search_qa(self, token_inflections, orig_tokenized, original_loss, question_dict, context):
        perturbed_tokenized = orig_tokenized.copy() # token list (list of str)

        max_loss = original_loss
        num_queries = 0
        max_predicted = ''

        detokenizer = MosesDetokenizer(lang='en')

        while True:
            new_tokenized_list = []
            # new_text_list = []
            new_loss_list = []
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
                    new_loss, new_predicted = self.get_loss(new_text, question_dict, context)
                    num_queries += 1

                    # record
                    new_tokenized_list.append(new_tokenized)
                    new_loss_list.append(new_loss)
                    new_predicted_list.append(new_predicted)

                # remove
                if perturbed_tokenized[position] != orig_tokenized[position]:
                    # do replace
                    new_tokenized = perturbed_tokenized.copy()
                    new_tokenized[position] = orig_tokenized[position]

                    # form text and eval
                    new_text = detokenizer.detokenize(new_tokenized)
                    new_loss, new_predicted = self.get_loss(new_text, question_dict, context)
                    num_queries += 1

                    # record
                    new_tokenized_list.append(new_tokenized)
                    new_loss_list.append(new_loss)
                    new_predicted_list.append(new_predicted)

            if len(new_loss_list) == 0: # no improve
                break

            cur_max_idx = np.argsort(new_loss_list)[-1]
            cur_max_loss = new_loss_list[cur_max_idx]
            cur_max_predicted = new_predicted_list[cur_max_idx]
            # cur_max_text = new_text_list[cur_max_idx]
            cur_max_tokenized = new_tokenized_list[cur_max_idx]

            # check stop criteria
            if metric_max_over_ground_truths(compute_f1, cur_max_predicted, question_dict['gold_texts']) == 0:
                perturbed_tokenized = cur_max_tokenized
                max_loss = cur_max_loss
                max_predicted = cur_max_predicted
                break

            if cur_max_loss > max_loss + EPSILON:
                perturbed_tokenized = cur_max_tokenized
                max_loss = cur_max_loss
                max_predicted = cur_max_predicted
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
            supp_loss, supp_predicted = self.get_loss(supp_text, question_dict, context)
            num_queries += 1

            if supp_loss > max_loss:
                max_loss = supp_loss
                max_predicted = supp_predicted
                perturbed_tokenized = supplement_tokenized

        return perturbed_tokenized, max_loss, max_predicted, num_queries

    @staticmethod
    def get_lowest_loss(start_logits_tensor, end_logits_tensor, gold_spans):
        start_logits_tensor = start_logits_tensor.to('cpu')
        end_logits_tensor = end_logits_tensor.to('cpu')

        target_tensors = [(torch.tensor([gold_start]), torch.tensor([gold_end])) for gold_start, gold_end in gold_spans]
        loss = CrossEntropyLoss()
        losses = []
        for target_start, target_end in target_tensors:
            avg_loss = (loss(start_logits_tensor, target_start) + loss(end_logits_tensor, target_end)) / 2
            losses.append(avg_loss)
        return min(losses).item()

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
class LSHuggingfaceBertQA(LocalSearchQA):
    def __init__(self, model_path, tokenizer_path, squad2=True, use_cuda=True):
        self.squad2 = squad2

        if torch.cuda.is_available() and use_cuda:
            self.device = 'cuda'
        else:
            self.device = 'cpu'
        config = AutoConfig.from_pretrained(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        self.model = AutoModelForQuestionAnswering.from_pretrained(model_path, config=config)
        self.num_special_tokens = len(self.tokenizer.build_inputs_with_special_tokens([], []))
        self.model.eval()
        self.model.to(self.device)

    def attack_one(self, question_dict, context, constrain_pos=True):
        if not self.squad2 and question_dict.get('is_impossible'):
            raise AssertionError('Not SQuAD 2 but has impossible questions')
        return super().attack_one(question_dict, context, constrain_pos)

    def get_loss(self, question, question_dict, context, max_seq_len=512, max_query_len=64):
        start_logits, end_logits, predicted = self.model_predict(question, context, max_seq_len, max_query_len)
        question_dict['gold_spans'] = list(set(
            self.get_gold_token_spans(self.tokenizer, question, question_dict, context, max_seq_len, max_query_len)))

        return super().get_lowest_loss(start_logits, end_logits, question_dict['gold_spans']), predicted

    @staticmethod
    def get_gold_token_spans(tokenizer, question, question_dict, context, max_seq_len, max_query_len):
        token_spans = []
        num_special_tokens = len(tokenizer.build_inputs_with_special_tokens([], []))

        if question_dict.get('is_impossible'):
            question_dict['gold_texts'] = ['']
            return [(0, 0)]

        qn_ids = tokenizer.encode(question, add_special_tokens=False)
        # context_ids = tokenizer.encode(context, add_special_tokens=False)
        # print(tokenizer.convert_ids_to_tokens(tokenizer.build_inputs_with_special_tokens(qn_ids, context_ids)))

        if len(qn_ids) > max_query_len:
            qn_ids = qn_ids[:max_query_len]

        for i, (char_span_start, char_span_end) in enumerate(question_dict['gold_char_spans']):
            gold_text = question_dict['gold_texts'][i]
            if gold_text == '.':
                # Ignore garbage annotation
                continue
            # Handle SentencePiece tokenizers
            if context[char_span_start - 1] == ' ' and len(
                    tokenizer.encode(context[:char_span_start], add_special_tokens=False)) != len(
                tokenizer.encode(context[:max(char_span_start - 1, 0)], add_special_tokens=False)):
                char_span_start -= 1
                gold_text = ' ' + gold_text

            span_start = len(qn_ids) + num_special_tokens - 1 + len(
                tokenizer.encode(context[:char_span_start], add_special_tokens=False))
            span_end = span_start + len(tokenizer.encode(gold_text, add_special_tokens=False))

            if span_start >= max_seq_len:
                span_start = 0
                span_end = 0
                question_dict['gold_texts'][i] = ''
            elif span_end >= max_seq_len:
                span_end = max_seq_len - 1
                print(span_end)
                context_ids = tokenizer.encode(context, add_special_tokens=False)
                max_context_len = max_seq_len - len(qn_ids) - num_special_tokens
                context_ids = context_ids[:max_context_len]
                tokens = tokenizer.convert_ids_to_tokens(
                    tokenizer.build_inputs_with_special_tokens(qn_ids, context_ids))
                question_dict['gold_texts'][i] = tokenizer.convert_tokens_to_string(
                    tokens[span_start:max_seq_len]).strip()

            token_spans.append((span_start, span_end))
        return token_spans

    def model_predict(self, question, context, max_seq_len=512, max_query_len=64):
        qn_ids = self.tokenizer.encode(question, add_special_tokens=False)
        if len(qn_ids) > max_query_len:
            qn_ids = qn_ids[:max_query_len]
        context_ids = self.tokenizer.encode(context, add_special_tokens=False)
        max_context_len = max_seq_len - len(qn_ids) - self.num_special_tokens - 1
        if len(context_ids) > max_context_len:
            context_ids = context_ids[:max_context_len]



        input_ids = self.tokenizer.build_inputs_with_special_tokens(qn_ids, context_ids)
        token_type_ids = self.tokenizer.create_token_type_ids_from_sequences(qn_ids, context_ids)
        with torch.no_grad():
             cur_output = self.model(torch.tensor([input_ids]).to(self.device), token_type_ids=torch.tensor([token_type_ids]).to(self.device))
             start_logits = cur_output[0]
             end_logits = cur_output[1]
            # start_logits, end_logits = self.model(torch.tensor([input_ids]).to(self.device),
            #                                       token_type_ids=torch.tensor([token_type_ids]).to(self.device))
        all_tokens = self.tokenizer.convert_ids_to_tokens(input_ids)
        predicted = self.tokenizer.convert_tokens_to_string(
                        all_tokens[torch.argmax(start_logits): torch.argmax(end_logits) + 1])
        if predicted == self.tokenizer.cls_token:
            predicted = ''
        return start_logits, end_logits, predicted


# class MorpheusBertQA(MorpheusQA):
#     def __init__(self, model_path, squad2=True, use_cuda=True):
#         self.squad2 = squad2
#         if torch.cuda.is_available() and use_cuda:
#             self.device = 'cuda'
#         else:
#             self.device = 'cpu'
#         config = BertConfig.from_pretrained(model_path + '/config.json')
#         self.tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
#         self.model = BertForQuestionAnswering.from_pretrained(model_path + '/pytorch_model.bin', config=config)
#
#         self.model.eval()
#         self.model.to(self.device)
#
#     def morph(self, question_dict, context, constrain_pos=True, conservative=False):
#         if not self.squad2 and question_dict.get('is_impossible'):
#             raise AssertionError('Not SQuAD 2 but has impossible questions')
#         return super().morph(question_dict, context, constrain_pos, conservative)
#
#     def get_loss(self, question, question_dict, context, max_seq_len=512, max_query_len=64):
#         start_logits, end_logits, predicted = self.model_predict(question, context, max_seq_len, max_query_len)
#         question_dict['gold_spans'] = list(set(
#             self.get_gold_token_spans(self.tokenizer, question, question_dict, context, max_seq_len, max_query_len)))
#
#         return super().get_lowest_loss(start_logits, end_logits, question_dict['gold_spans']), predicted
#
#     @staticmethod
#     def get_gold_token_spans(tokenizer, question, question_dict, context, max_seq_len, max_query_len):
#         token_spans = []
#
#         if question_dict.get('is_impossible'):
#             question_dict['gold_texts'] = ['']
#             return [(0, 0)]
#
#         qn_ids = tokenizer.encode(question, add_special_tokens=False)
#         if len(qn_ids) > max_query_len:
#             qn_ids = qn_ids[:max_query_len]
#
#         for i, (char_span_start, char_span_end) in enumerate(question_dict['gold_char_spans']):
#
#             span_start = len(qn_ids) + 2 + len(tokenizer.encode(context[:char_span_start], add_special_tokens=False))
#             span_end = span_start + len(tokenizer.encode(question_dict['gold_texts'][i], add_special_tokens=False)) - 1
#
#             if span_start >= max_seq_len:
#                 span_start = 0
#                 span_end = 0
#                 question_dict['gold_texts'][i] = ''
#             elif span_end >= max_seq_len:
#                 span_end = max_seq_len - 1
#                 context_ids = tokenizer.encode(context, add_special_tokens=False)
#                 max_context_len = max_seq_len - len(qn_ids) - 3
#                 context_ids = context_ids[:max_context_len]
#                 tokens = tokenizer.convert_ids_to_tokens(
#                     tokenizer.build_inputs_with_special_tokens(qn_ids, context_ids))
#                 question_dict['gold_texts'][i] = tokenizer.convert_tokens_to_string(tokens[span_start:max_seq_len])
#
#             token_spans.append((span_start, span_end))
#         return token_spans
#
#     def model_predict(self, question, context, max_seq_len=512, max_query_len=64):
#         qn_ids = self.tokenizer.encode(question, add_special_tokens=False)
#         if len(qn_ids) > max_query_len:
#             qn_ids = qn_ids[:max_query_len]
#         context_ids = self.tokenizer.encode(context, add_special_tokens=False)
#         max_context_len = max_seq_len - len(qn_ids) - 3
#         if len(context_ids) > max_context_len:
#             context_ids = context_ids[:max_context_len]
#
#         input_ids = self.tokenizer.build_inputs_with_special_tokens(qn_ids, context_ids)
#         token_type_ids = self.tokenizer.create_token_type_ids_from_sequences(qn_ids, context_ids)
#         with torch.no_grad():
#             start_logits, end_logits = self.model(torch.tensor([input_ids]).to(self.device),
#                                                   token_type_ids=torch.tensor([token_type_ids]).to(self.device))
#         all_tokens = self.tokenizer.convert_ids_to_tokens(input_ids)
#         predicted = self.tokenizer.convert_tokens_to_string(
#             all_tokens[torch.argmax(start_logits): torch.argmax(end_logits) + 1])
#         if predicted == '[CLS]':
#             predicted = ''
#         return start_logits, end_logits, predicted


class LSBidafQA(LocalSearchQA):
    def __init__(self, model_path, use_cuda=True, cuda_device=0):
        if model_path == 'bidaf':
            model_path = "https://allennlp.s3.amazonaws.com/models/bidaf-model-2017.09.15-charpad.tar.gz"
        elif model_path == 'elmobidaf':
            model_path = "https://s3.us-west-2.amazonaws.com/allennlp/models/bidaf-elmo-model-2018.11.30-charpad.tar.gz"

        if not torch.cuda.is_available() or not use_cuda:
            cuda_device = -1

        self.model = Predictor.from_path(model_path, cuda_device=cuda_device)

    def attack_one(self, question_dict, context, constrain_pos=False):
        if question_dict.get('is_impossible'):
            raise AssertionError('BiDAF only attack answerable question')
        return super().attack_one(question_dict, context, constrain_pos)

    def get_loss(self, question, question_dict, context):
        predicted = self.model.predict(passage=context, question=question)

        start_logits_tensor = torch.tensor(predicted['span_start_logits']).unsqueeze(0)
        end_logits_tensor = torch.tensor(predicted['span_end_logits']).unsqueeze(0)

        question_dict['gold_spans'] = self.get_gold_token_spans(WordTokenizer(), question_dict['gold_char_spans'],
                                                                context)
        return super().get_lowest_loss(start_logits_tensor, end_logits_tensor, question_dict['gold_spans']), predicted[
            'best_span_str']

    def model_predict(self, question, context):
        predicted = self.model.predict(passage=context, question=question)

        return predicted['span_start_logits'], predicted['span_end_logits'], predicted['best_span_str']

    @staticmethod
    def get_gold_token_spans(tokenizer, gold_char_spans, context):
        # Adapted from AllenNLP
        passage_tokens = tokenizer.tokenize(context)
        token_spans = []
        passage_offsets = [(token.idx, token.idx + len(token.text)) for token in passage_tokens]

        for char_span_start, char_span_end in gold_char_spans:
            if char_span_end > passage_offsets[-1][1]:
                continue
            (span_start, span_end), error = char_span_to_token_span(passage_offsets, (char_span_start, char_span_end))
            token_spans.append((span_start, span_end))
        return token_spans

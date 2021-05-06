# coding: utf-8
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import sys
import attr
import copy
from SbGS_related.config import config
import nltk
import spacy
from functools import partial
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
# nltk.download('wordnet')
from nltk.corpus import wordnet as wn
from SbGS_related.get_NE_list import NE_list
import numpy as np
from SbGS_related.unbuffered import Unbuffered

from SbGS_related.PWWS_paras import EPSILON

sys.stdout = Unbuffered(sys.stdout)
# from pywsd.lesk import simple_lesk as disambiguate

nlp = spacy.load('en_core_web_sm')
# nlp2 = spacy.load('en', tagger=False, entity=False)
# Penn TreeBank POS tags:
# http://www.ling.upenn.edu/courses/Fall_2003/ling001/penn_treebank_pos.html

supported_pos_tags = [
    'CC',  # coordinating conjunction, like "and but neither versus whether yet so"
    # 'CD',   # Cardinal number, like "mid-1890 34 forty-two million dozen"
    # 'DT',   # Determiner, like all "an both those"
    # 'EX',   # Existential there, like "there"
    # 'FW',   # Foreign word
    # 'IN',   # Preposition or subordinating conjunction, like "among below into"
    'JJ',  # Adjective, like "second ill-mannered"
    'JJR',  # Adjective, comparative, like "colder"
    'JJS',  # Adjective, superlative, like "cheapest"
    # 'LS',   # List item marker, like "A B C D"
    # 'MD',   # Modal, like "can must shouldn't"
    'NN',  # Noun, singular or mass
    'NNS',  # Noun, plural
    'NNP',  # Proper noun, singular
    'NNPS',  # Proper noun, plural
    # 'PDT',  # Predeterminer, like "all both many"
    # 'POS',  # Possessive ending, like "'s"
    # 'PRP',  # Personal pronoun, like "hers herself ours they theirs"
    # 'PRP$',  # Possessive pronoun, like "hers his mine ours"
    'RB',  # Adverb
    'RBR',  # Adverb, comparative, like "lower heavier"
    'RBS',  # Adverb, superlative, like "best biggest"
    # 'RP',   # Particle, like "board about across around"
    # 'SYM',  # Symbol
    # 'TO',   # to
    # 'UH',   # Interjection, like "wow goody"
    'VB',  # Verb, base form
    'VBD',  # Verb, past tense
    'VBG',  # Verb, gerund or present participle
    'VBN',  # Verb, past participle
    'VBP',  # Verb, non-3rd person singular present
    'VBZ',  # Verb, 3rd person singular present
    # 'WDT',  # Wh-determiner, like "that what whatever which whichever"
    # 'WP',   # Wh-pronoun, like "that who"
    # 'WP$',  # Possessive wh-pronoun, like "whose"
    # 'WRB',  # Wh-adverb, like "however wherever whenever"
]

unsupported_NE_pos = {
  "-LRB-", "-RRB-", ".", ":", "``", ",",  "''", "HYPH", "LS", "NFP", "_SP", # marks and Puncutation
  "DT",
  "POS",
  "TO",
  "PRP",
  }

# @attr.s
# class SubstitutionCandidate:
#     token_position = attr.ib()
#     similarity_rank = attr.ib()
#     original_token = attr.ib()
#     candidate_word = attr.ib()


def vsm_similarity(doc, original, synonym):
    window_size = 3
    start = max(0, original.i - window_size)
    return doc[start: original.i + window_size].similarity(synonym)


def _get_wordnet_pos(spacy_token):
    '''Wordnet POS tag'''
    pos = spacy_token.tag_[0].lower()
    if pos in ['r', 'n', 'v']:  # adv, noun, verb
        return pos
    elif pos == 'j':
        return 'a'  # adj


def _synonym_prefilter_fn(token, synonym):
    '''
    Similarity heuristics go here
    '''
    if (len(synonym.text.split()) > 2 or (  # the synonym produced is a phrase
            synonym.lemma == token.lemma) or (  # token and synonym are the same
            synonym.tag != token.tag) or (  # the pos of the token synonyms are different
            token.text.lower() == 'be')):  # token is be
        return False
    else:
        return True


def _generate_synonym_candidates(token, token_position, rank_fn=None):
    '''
    Generate synonym candidates.
    For each token in the doc, the list of WordNet synonyms is expanded.
    :return candidates, a list, whose type of element is <class '__main__.SubstitutionCandidate'>
            like SubstitutionCandidate(token_position=0, similarity_rank=10, original_token=Soft, candidate_word='subdued')
    '''
    candidates = []
    if token.tag_ in supported_pos_tags:
        wordnet_pos = _get_wordnet_pos(token)  # 'r', 'a', 'n', 'v' or None
        wordnet_synonyms = []

        synsets = wn.synsets(token.text, pos=wordnet_pos)
        for synset in synsets:
            wordnet_synonyms.extend(synset.lemmas())

        synonyms = []
        for wordnet_synonym in wordnet_synonyms:
            spacy_synonym = nlp(wordnet_synonym.name().replace('_', ' '))[0]
            synonyms.append(spacy_synonym)

        synonyms = filter(partial(_synonym_prefilter_fn, token), synonyms)

        candidate_set = set()
        for _, synonym in enumerate(synonyms):
            candidate_word = synonym.text
            if candidate_word in candidate_set:  # avoid repetition
                continue
            candidate_set.add(candidate_word)
            # candidate = SubstitutionCandidate(
            #     token_position=token_position,
            #     similarity_rank=None,
            #     original_token=token,
            #     candidate_word=candidate_word)
            candidates.append(candidate_word)
    return candidates


# def _compile_perturbed_tokens(doc, accepted_candidates):
#     '''
#     Traverse the list of accepted candidates and do the token substitutions.
#     '''
#     candidate_by_position = {}
#     for candidate in accepted_candidates:
#         candidate_by_position[candidate.token_position] = candidate
#
#     final_tokens = []
#     for position, token in enumerate(doc):
#         word = token.text
#         if position in candidate_by_position:
#             candidate = candidate_by_position[position]
#             word = candidate.candidate_word.replace('_', ' ')
#         final_tokens.append(word)
#
#     return final_tokens

def _do_replace(word_list, position, new_word):
    new_word_list = word_list.copy()
    new_word_list[position] = new_word
    return new_word_list



def LS(
        word_list,
        true_y,
        dataset,
        position_word_list=None,
        rank_fn=None,
        heuristic_fn=None,  # Defined in adversarial_tools.py
        halt_condition_fn=None,  # Defined in adversarial_tools.py
        verbose=True):


    heuristic_fn = heuristic_fn
    halt_condition_fn = halt_condition_fn
    # perturbed_doc = doc
    # perturbed_text = perturbed_doc.text

    orig_word_list = word_list.copy()
    orig_text = " ".join(word_list)
    best_score = heuristic_fn(orig_text)

    NE_candidates = NE_list.L[dataset][true_y]

    NE_tags = list(NE_candidates.keys())
    use_NE = True  # whether use NE as a substitute

    max_len = config.word_max_len[dataset]

    def get_modif_rate(ret_word_list):
        substitute_count = 0
        assert len(ret_word_list) == len(orig_word_list)
        for position in range(len(orig_word_list)):
            if orig_word_list[position] != ret_word_list[position]:
                print(f'position {position}, raw {orig_word_list[position]} -> new {ret_word_list[position]}')
                substitute_count += 1
        modif_rate = substitute_count / len(orig_word_list)
        return modif_rate

    # ================ Generate Candidates ===============================
    # for each word w_i in x, use WordNet to build a synonym set L_i
    candidates_by_position = {}
    for (position, token) in position_word_list:
        if position >= max_len:
            break

        candidates = []
        if use_NE:
            NER_tag = token.ent_type_
            if NER_tag in NE_tags and token.tag_ not in unsupported_NE_pos: #  and token.tag_ != 'DT'
                # candidate = SubstitutionCandidate(position, 0, token, NE_candidates[NER_tag])
                candidate = NE_candidates[NER_tag]
                candidates.append(candidate)
            else:
                candidates = _generate_synonym_candidates(token=token, token_position=position, rank_fn=rank_fn)
        else:
            candidates = _generate_synonym_candidates(token=token, token_position=position, rank_fn=rank_fn)

        if len(candidates) == 0:
            continue
        cur_orig_word = orig_word_list[position]
        candidates.append(cur_orig_word)
        candidates_by_position[position] = candidates

    # local search
    adv_word_list = word_list.copy()
    while True:

        new_score_list = []
        new_word_lists = []
        new_text_list = []
        # tmp_change_list = []

        for position in candidates_by_position:
            candidates = candidates_by_position[position]
            for candidate_w in candidates:
                if adv_word_list[position] == candidate_w:
                    continue
                else:
                    new_word_list = _do_replace(adv_word_list, position, candidate_w)
                    new_text = " ".join(new_word_list)
                    new_score = heuristic_fn(new_text)

                    new_word_lists.append(new_word_list)
                    new_score_list.append(new_score)
                    new_text_list.append(new_text)
                    # tmp_change_list.append(candidate_w)

        # check
        if len(new_score_list) == 0:
            return orig_text, 0

        # get current best
        cur_best_idx = np.argsort(new_score_list)[-1]
        cur_best_score = new_score_list[cur_best_idx]
        cur_best_word_list = new_word_lists[cur_best_idx]
        cur_best_text = new_text_list[cur_best_idx]

        # print('='*20)
        # print(f'best position: {tmp_change_list[cur_best_idx].token_position},'
        #       f' best word {tmp_change_list[cur_best_idx].candidate_word}',
        #       f' raw1: {tmp_change_list[cur_best_idx].original_token.text}')
        #
        # get_modif_rate(cur_best_text)
        # print('=' * 20)

        # check finish
        if halt_condition_fn(cur_best_text):
            sub_rate = get_modif_rate(cur_best_word_list)
            print('Success in LS')
            return cur_best_text, sub_rate

        # check best
        if cur_best_score > best_score + EPSILON:
            adv_word_list = cur_best_word_list
            best_score = cur_best_score
        else:
            break

    # check supplement:
    supplement_candidates_by_position = {}
    for position in candidates_by_position:

        supplement_candidates = []
        if len(candidates_by_position[position]) > 0:
            for candidate_w in candidates_by_position[position]:
                if candidate_w != adv_word_list[position] and candidate_w != orig_word_list[position]:
                    supplement_candidates.append(candidate_w)

        if len(supplement_candidates) == 0:
            continue
        supplement_candidates_by_position[position] = supplement_candidates

    # check validation of supplementary set
    is_set_valid = True
    for position in supplement_candidates_by_position:
        supplement_candidates = supplement_candidates_by_position[position]
        if len(supplement_candidates) != 1:
            is_set_valid = False
            break

    if is_set_valid:
        # check supplement
        supp_word_list = orig_word_list.copy()
        for position in supplement_candidates_by_position:
            supp_word_list[position] = supplement_candidates_by_position[position][0]

        supplement_text = ' '.join(supp_word_list)
        if halt_condition_fn(supplement_text):
            modif_rate = get_modif_rate(supp_word_list)
            print('Succsess in sup')
            return supplement_text, modif_rate

    print('Fail')
    return orig_text, 0



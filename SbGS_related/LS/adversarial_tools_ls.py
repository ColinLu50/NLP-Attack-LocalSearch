import sys

import spacy

from SbGS_related.LS.paraphrase_ls import LS
from SbGS_related.char_level_process import doc_process, get_embedding_dict
from SbGS_related.config import config
from SbGS_related.unbuffered import Unbuffered
from SbGS_related.word_level_process import text_to_vector

sys.stdout = Unbuffered(sys.stdout)
# nlp = spacy.load('en', tagger=False, entity=False)
nlp = spacy.load('en_core_web_sm')

def get_word_position(doc, dataset, level):
    position_word_list = []
    word_list = []

    if level == 'word':
        max_len = config.word_max_len[dataset]
        # origin_vector = text_to_vector(text, tokenizer, dataset)
        for position in range(len(doc)):
            if position >= max_len:
                break
            # get x_i^(\hat)
            position_word_list.append((position, doc[position]))
    elif level == 'char':
        max_len = config.char_max_len[dataset]
        find_a_word = False
        word_position = 0
        for i, c in enumerate(doc.text):
            if i >= max_len:
                break

            if c is not ' ':
                pass
            else:
                find_a_word = True
                position_word_list.append((word_position, doc[word_position]))
                word_position += 1

            if find_a_word:
                find_a_word = False

    for position, token in position_word_list:
        word_list.append(token.text)

    return position_word_list, word_list


def adversarial_paraphrase_LS(input_text, true_y, grad_guide, tokenizer, dataset, level, verbose=True):
    '''
    Compute a perturbation, greedily choosing the synonym if it causes the most
    significant change in the classification probability after replacement
    :return perturbed_text: generated adversarial examples
    :return perturbed_y: predicted class of perturbed_text
    :return sub_rate: word replacement rate showed in Table 3
    :return change_tuple_list: list of substitute words
    '''

    def halt_condition_fn(perturbed_text):
        '''
        Halt if model output is changed.
        '''
        perturbed_vector = None
        if level == 'word':
            perturbed_vector = text_to_vector(perturbed_text, tokenizer, dataset)
        elif level == 'char':
            max_len = config.char_max_len[dataset]
            perturbed_vector = doc_process(perturbed_text, get_embedding_dict(), dataset).reshape(1, max_len)
        adv_y = grad_guide.predict_classes(input_vector=perturbed_vector)
        if adv_y != true_y:
            return True
        else:
            return False

    def heuristic_fn(perturbed_text):
        '''
        Return the difference between the classification probability of the original
        word and the candidate substitute synonym, which is defined in Eq.(4) and Eq.(5).
        '''
        # doc = nlp(text)
        perturbed_vector = None

        if level == 'word':
            perturbed_vector = text_to_vector(perturbed_text, tokenizer, dataset)
        elif level == 'char':
            max_len = config.char_max_len[dataset]
            perturbed_vector = doc_process(perturbed_text, get_embedding_dict(), dataset).reshape(1, max_len)

        perturbed_prob = grad_guide.predict_prob(input_vector=perturbed_vector)
        untarget_score = 1 - perturbed_prob[true_y]

        return untarget_score

    doc_init = nlp(input_text)
    # form std text
    std_token_list = []
    for position, token in enumerate(doc_init):
        word = token.text
        if word.replace(" ", "") == "":
            continue
        std_token_list.append(word)
    input_text = " ".join(std_token_list)
    doc = nlp(input_text)



    # local search
    position_word_list, word_list = get_word_position(doc, dataset, level)

    if level == 'char':
        # char cnn need to keep raw sentence to reconstruct the raw sentence
        word_list = std_token_list


    perturbed_text, modif_rate = LS(word_list,
                                    true_y,
                                    dataset,
                                    position_word_list=position_word_list,
                                    heuristic_fn=heuristic_fn,
                                    halt_condition_fn=halt_condition_fn,
                                    verbose=verbose)

    # print("perturbed_text after perturb_text:", perturbed_text)
    origin_vector = perturbed_vector = None
    if level == 'word':
        origin_vector = text_to_vector(input_text, tokenizer, dataset)
        perturbed_vector = text_to_vector(perturbed_text, tokenizer, dataset)
    elif level == 'char':
        max_len = config.char_max_len[dataset]
        origin_vector = doc_process(input_text, get_embedding_dict(), dataset).reshape(1, max_len)
        perturbed_vector = doc_process(perturbed_text, get_embedding_dict(), dataset).reshape(1, max_len)
    perturbed_y = grad_guide.predict_classes(input_vector=perturbed_vector)
    if verbose:
        origin_prob = grad_guide.predict_prob(input_vector=origin_vector)
        perturbed_prob = grad_guide.predict_prob(input_vector=perturbed_vector)
        raw_score = origin_prob[true_y] - perturbed_prob[true_y]
        print('Prob before: ', origin_prob[true_y], '. Prob after: ', perturbed_prob[true_y],
              '. Prob shift: ', raw_score)
    return perturbed_text, perturbed_y, modif_rate

import sys
import spacy
from SbGS_related.config import config
from SbGS_related.PWWS.paraphrase import _compile_perturbed_tokens, PWWS
from SbGS_related.word_level_process import text_to_vector
from SbGS_related.char_level_process import doc_process, get_embedding_dict
from SbGS_related.PWWS.evaluate_word_saliency import evaluate_word_saliency
# from keras.backend.tensorflow_backend import set_session
from SbGS_related.unbuffered import Unbuffered

sys.stdout = Unbuffered(sys.stdout)
# nlp = spacy.load('en', tagger=False, entity=False)
nlp = spacy.load('en_core_web_sm')





def adversarial_paraphrase(input_text, true_y, grad_guide, tokenizer, dataset, level, verbose=True):
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

    def heuristic_fn(text, candidate):
        '''
        Return the difference between the classification probability of the original
        word and the candidate substitute synonym, which is defined in Eq.(4) and Eq.(5).
        '''
        doc = nlp(text)
        origin_vector = None
        perturbed_vector = None
        if level == 'word':
            origin_vector = text_to_vector(text, tokenizer, dataset)
            perturbed_tokens = _compile_perturbed_tokens(doc, [candidate])
            perturbed_doc = nlp(' '.join(perturbed_tokens))
            perturbed_vector = text_to_vector(perturbed_doc.text, tokenizer, dataset)
        elif level == 'char':
            max_len = config.char_max_len[dataset]
            origin_vector = doc_process(text, get_embedding_dict(), dataset).reshape(1, max_len)
            perturbed_tokens = _compile_perturbed_tokens(nlp(input_text), [candidate])
            perturbed_text = ' '.join(perturbed_tokens)
            perturbed_vector = doc_process(perturbed_text, get_embedding_dict(), dataset).reshape(1, max_len)

        origin_prob = grad_guide.predict_prob(input_vector=origin_vector)
        perturbed_prob = grad_guide.predict_prob(input_vector=perturbed_vector)
        delta_p = origin_prob[true_y] - perturbed_prob[true_y]

        return delta_p

    doc = nlp(input_text)
    # Bug: need to make the format of the input text same as that of output text
    # Remove empty token text
    std_tokens = []
    for position, token in enumerate(doc):
        word = token.text
        if word.replace(" ", "") == "":
            continue
        std_tokens.append(word)
    input_text = ' '.join(std_tokens)
    doc = nlp(input_text)


    # PWWS
    position_word_list, word_saliency_list = evaluate_word_saliency(doc, grad_guide, tokenizer, true_y, dataset, level)
    perturbed_text, sub_rate, NE_rate, change_tuple_list = PWWS(doc,
                                                                true_y,
                                                                dataset,
                                                                word_saliency_list=word_saliency_list,
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
    return perturbed_text, perturbed_y, sub_rate, NE_rate, change_tuple_list
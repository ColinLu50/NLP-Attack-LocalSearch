from abc import ABCMeta, abstractmethod
import lemminflect, random


class LocalSearchBase(metaclass=ABCMeta):
    @abstractmethod
    def attack_one(self):
         pass
    @staticmethod
    def get_inflections(orig_tokenized, pos_tagged, constrain_pos):
        have_inflections = {'NOUN', 'VERB', 'ADJ'}
        token_inflections = [] # elements of form (i, inflections) where i is the token's position in the sequence

        for i, word in enumerate(orig_tokenized):
            lemmas = lemminflect.getAllLemmas(word)
            if lemmas and pos_tagged[i][1] in have_inflections:
                if pos_tagged[i][1] in lemmas:
                    lemma = lemmas[pos_tagged[i][1]][0]
                else:
                    lemma = random.choice(list(lemmas.values()))[0]

                if constrain_pos:
                    inflections = (i, list(set([infl for tup in lemminflect.getAllInflections(lemma, upos=pos_tagged[i][1]).values() for infl in tup])))
                else:
                    inflections = (i, list(set([infl for tup in lemminflect.getAllInflections(lemma).values() for infl in tup])))

                random.shuffle(inflections[1])
                token_inflections.append(inflections)
        return token_inflections
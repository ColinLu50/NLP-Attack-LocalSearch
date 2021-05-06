import numpy as np
from keras import backend as K




class ForwardGradWrapper:
    '''
    Utility class that computes the classification probability of model input and predict its class
    '''

    def __init__(self, model):
        '''
        :param model: Keras model.
            This code makes a bunch of assumptions about the model:
            - Model has single input
            - Embedding is the first layer
            - Model output is a scalar (logistic regression)
        '''
        input_tensor = model.input

        self.model = model
        self.input_tensor = input_tensor
        self.sess = K.get_session()

        self.in_attack = False
        self.query_num = 0

    def predict_prob(self, input_vector):
        if self.in_attack:
            self.query_num += input_vector.shape[0]
        prob = self.model.predict(input_vector).squeeze()
        return prob

    def predict_classes(self, input_vector):
        if self.in_attack:
            self.query_num += input_vector.shape[0]

        prediction = self.model.predict(input_vector)
        classes = np.argmax(prediction, axis=1)

        return classes

    def start_attack(self):
        self.in_attack = True
        self.query_num = 0

    def end_attack(self):
        self.in_attack = False

        return self.query_num

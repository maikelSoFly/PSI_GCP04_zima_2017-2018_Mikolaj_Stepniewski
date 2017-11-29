import numpy as np
import random

""" Sign function which can be translated by given value. Used as
    activation function for perceptron.

    - Parameters:
        - translation: breaking point for the function.

    - Usage:
        - Sign()(0.5)
            - returns sign function for unipolar sigmoidal function
"""
class Sign:
    def __call__(self, translation):
        def sign(x):
            if x < translation:
                return 0
            else:
                return 1
        return sign

    def derivative(self):
        def signDeriv(x):
            return 1
        return signDeriv


""" Sigmoidal function & its derivative for given beta. Used as
    Activation function for perceptron.

    - Parameters:
        - beta: sigmoidal function parameter. Its value affects
        function shape. The greater the value the steeper is the function.

    - Usage:
        - Sigm()(0.5)
            - returns: sigm(x) function with beta=0.5
        - Sigm().derivative(0.5)
            - returns sigmDeriv function
            with beta=0.5
"""
class Sigm:
    def __call__(self, beta):
        def sigm(x):
            return 1.0 / (1.0 + np.exp(-beta * x))
        sigm.__name__ += '({0:.3f})'.format(beta)
        return sigm

    def derivative(self, beta):
        def sigmDeriv(x):
            return beta * np.exp(-beta * x) / ((1.0 + np.exp(-beta * x))**2)
        sigmDeriv.__name__ += '({0:.3f})'.format(beta)
        return sigmDeriv

class Linear:
    def __call__(self):
        def linear(x):
            return x
        return linear
    def derivative(self):
        def linearDeriv(x):
            return 1
        return linearDeriv


class HebbNeuron:
    def __init__(self, weights, iid, activFunc, activFuncDeriv, lRate=0.01, bias=random.uniform(-1, 1)):
        self.__dict__['_weights'] = np.array(weights)
        self.__dict__['_activFunc'] = activFunc
        self.__dict__['_activFuncDeriv'] = activFuncDeriv
        self.__dict__['_bias'] = bias
        self.__dict__['_lRate'] = lRate
        self.__dict__['_inputValues'] = None
        self.__dict__['_error'] = None
        self.__dict__['_sum'] = None
        self.__dict__['_val'] = None
        self.__dict__['_iid'] = iid
        self.__dict__['_delta'] = []

    def process(self, inputs):
        self._inputValues = inputs
        self._sum = np.dot(self._weights, np.array(self._inputs)) + self._bias
        self._val = self._activFunc(self._sum)
        return self._val

    def train(self, inputs):
        self._inputValues = inputs
        self._sum = np.dot(self._weights, np.array(self._inputs)) + self._bias
        self._val = self._activFunc(self._sum)

        for i in range(len(self._weights)):
            self._weights[i] += self._inputs[i] * self._val * self._lRate

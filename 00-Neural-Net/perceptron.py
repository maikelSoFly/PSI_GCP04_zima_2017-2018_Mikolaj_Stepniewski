import random
import numpy as np

def sign(x):
    if x >= 0:
        return 1
    else:
        return -1

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
            return 1.0/(1.0+np.exp(-beta*x))
        sigm.__name__ += '({0:.3f})'.format(beta)
        return sigm
    def derivative(self, beta):
        def sigmDeriv(x):
            return beta*np.exp(-beta*x)/((1.0+np.exp(-beta*x))**2)
        sigmDeriv.__name__ += '({0:.3f})'.format(beta)
        return sigmDeriv



class Perceptron:
    """ Perceptron is a simple neural net that can
        specify which class object belongs to. """

    def __init__(self, weights, activFunc, activFuncDeriv, bias=-0.8*np.random.ranf()-0.1, lRate=0.5):
        self.__dict__['_weights'] = np.array(weights)
        self.__dict__['_activFunc'] = activFunc
        self.__dict__['_activFuncDeriv'] = activFuncDeriv
        self.__dict__['_bias'] = bias
        self.__dict__['_lRate'] = lRate
        self.__dict__['_inputValues']=None
        self.__dict__['_error']=None
        self.__dict__['_sum']=None

    def process(self, input):
        self._inputValues = np.array(input)
        for i in range(len(self._weights)):
            self._sum = np.dot(self._inputValues, self._weights) + self._bias

        """ Process output """
        return self._activFunc(self._sum)

    def train(self, input, target):
        guess = self.process(input)
        delta = guess - target
        self._error = delta * self._activFuncDeriv(self._sum)

        for i in range(len(self._weights)):
            self._weights[i] -= self._lRate * self._error * input[i]

        self._bias = self._lRate * self._error

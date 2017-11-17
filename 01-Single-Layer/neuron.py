import numpy as np
import random
from enum import Enum


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
            if x >= translation:
                return 1
            else:
                return 0
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


class Neuron:
    """ This is template class for both perceptron and sigmoidal neuron. Perceptron is able
    to specify which class object belongs to, returning only 0 or 1, whereas sigmoidal neuron can return
    every value from 0 to 1. """

    def __init__(self, weights, activFunc, activFuncDeriv, lRate=0.05, bias=random.uniform(-1, 1)):
        self.__dict__['_weights'] = np.array(weights)
        self.__dict__['_activFunc'] = activFunc
        self.__dict__['_activFuncDeriv'] = activFuncDeriv
        self.__dict__['_bias'] = bias
        self.__dict__['_lRate'] = lRate
        self.__dict__['_inputValues'] = None
        self.__dict__['_error'] = None
        self.__dict__['_sum'] = None
        self.__dict__['_val'] = None

    def process(self, input):
        self._inputValues = np.array(input)
        self._sum = np.dot(self._inputValues, self._weights) + self._bias

        """ Process output """
        self._val = self._activFunc(self._sum)
        return self._val

    def train(self, input, target):
        guess = self.process(input)
        delta = guess - target

        """ Updating weights based on error.
            Gradient learning """
        self._error = self._lRate * delta * self._activFuncDeriv(self._sum)

        for i in range(len(self._weights)):
            self._weights[i] -=  * self._error * input[i]

        self._bias = self._lRate * self._error

    """ Access method """
    def __getitem__(self, index):
        if index == 'val':
            return self._val
        elif index == 'sum':
            return self._sum
        elif index == 'error':
            return self._error


class Layer:
    def __init__(self, numOfNeurons, numOfInputs, activFunc, activFuncDeriv):
        self.__dict__['_neurons'] = []
        self.__dict__['_numOfNeurons'] = numOfNeurons
        self.__dict__['_activFunc'] = activFunc
        self.__dict__['_activFuncDeriv'] = activFuncDeriv
        self.__dict__['_numOfInputs'] = numOfInputs

        """ Creating neurons """
        for n in range(numOfNeurons):
            w = [random.uniform(-1, 1) for _ in range(numOfInputs)]
            self._neurons.append(Neuron(w, activFunc, activFuncDeriv))

    def processNeurons(self, inputs):
        """ Passing data through the neurons of the layer.
        Used for validation. """
        outputs = []
        for n in self._neurons:
            outputs.append(n.process(inputs))
        return outputs

    def trainNeurons(self, inputs, desired):
        """ Passing training data through neurons of the layer. """
        outputs = []
        for index, n in enumerate(self._neurons):
            if self._numOfNeurons > 1:
                n.train(inputs, desired[index])
            else:
                n.train(inputs, desired)
            outputs.append(n._val)
        return outputs


class LayerManager:
    """ Class which manages single layer of neurons and Perceptron (processing
    outputs of the layer). """
    def __init__(self, numOfLayers, numOfNeurons, numOfInputs, activFuncs, activFuncDerivs):
        self.__dict__['_layers'] = []
        self.__dict__['_numOfLayers'] = numOfLayers
        self.__dict__['_numOfInputs'] = numOfInputs
        self.__dict__['_activFuncs'] = activFuncs
        self.__dict__['_activFuncDerivs'] = activFuncDerivs

        """ Creating single layers """
        for i in range(numOfLayers):
            self._layers.append(
                Layer(numOfNeurons[i], numOfInputs[i], activFuncs[i], activFuncDerivs[i]))

    def processLayers(self, inputs):
        """ Passing data through layers.
        Used for validation. """
        prevOuts = None
        output = []
        for i in range(self._numOfLayers):
            if i == 0:
                prevOuts = self._layers[i].processNeurons(inputs)
                output.append(prevOuts)
            else:
                prevOuts = self._layers[i].processNeurons(prevOuts)
                output.append(prevOuts)
        return output

    def trainLayers(self, inputVectors):
        """ Passing training data through layers. """
        results = []
        for i in range(self._numOfLayers):
            results.append(self._layers[i].trainNeurons(
                inputVectors[i]._x, inputVectors[i]._d))
        return results

    """ Access method """
    def __getitem__(self, index):
        if index == 'layers':
            return self._layers
        elif index == 'numOfLayers':
            return self._numOfLayers

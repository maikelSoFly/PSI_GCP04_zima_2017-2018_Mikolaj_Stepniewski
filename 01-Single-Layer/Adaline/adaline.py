import random
import numpy as np
from enum import Enum
import time


""" Sign function which can be translated by given value. Used as
    activation function for perceptron.
    - Parameters:
        - translation: breaking point for the function.
    - Usage:
        - Sign()(n - 1)
            - returns sign function with threshold for AND type, where n
            is number of inputs
"""


class Sign:
    def __call__(self, translation):
        def sign(x):
            if x > translation:
                return 1.0
            else:
                return -1.0
        return sign


class Adaline:
    def __init__(self, weights, activFunc, lRate=0.1, bias=random.uniform(-1, 1)):
        self.__dict__['_weights'] = weights
        self.__dict__['_bias'] = bias
        self.__dict__['_lRate'] = lRate
        self.__dict__['_activFunc'] = activFunc
        self.__dict__['_sum'] = None
        self.__dict__['_val'] = None
        self.__dict__['_inputs'] = None

    def process(self, inputs):
        self._inputs = np.array(inputs)
        self._sum = np.dot(self._inputs, self._weights) + self._bias
        self._val = self._activFunc(self._sum)
        return self._val

    def train(self, inputs, desired):
        guess = self.process(inputs)
        error = desired - self._sum

        for i in range(len(self._weights)):
            self._weights[i] += self._lRate * error * self._inputs[i]

        self._bias = self._lRate * error

    def __getitem__(self, index):
        return self._weights[index]

    def __len__(self):
        return len(self._weights)


class Layer:
    def __init__(self, numOfNeurons, numOfInputs, activFunc):
        self.__dict__['_neurons'] = []
        self.__dict__['_numOfNeurons'] = numOfNeurons
        self.__dict__['_activFunc'] = activFunc
        self.__dict__['_numOfInputs'] = numOfInputs

        for n in range(numOfNeurons):
            w = [random.uniform(-1, 1) for _ in range(numOfInputs)]
            self._neurons.append(Adaline(w, activFunc))

    def processNeurons(self, inputs):
        outputs = []
        for n in self._neurons:
            outputs.append(n.process(inputs))
        return outputs

    def trainNeurons(self, inputs, desired):
        outputs = []
        for index, n in enumerate(self._neurons):
            if self._numOfNeurons > 1:
                n.train(inputs, desired[index])
            else:
                n.train(inputs, desired)
            outputs.append(n._val)
        return outputs


class Madaline:
    def __init__(self, numOfNeurons, numOfInputs, activFunc, thresholdFuncType):
        self.__dict__['_numOfNeurons'] = numOfNeurons
        self.__dict__['_activFunc'] = activFunc
        self.__dict__['_thresholdFuncType'] = thresholdFuncType
        self.__dict__['_thresholdFunc'] = self.getThresholdFunc(
            numOfInputs, thresholdFuncType)
        self.__dict__['_numOfInputs'] = numOfInputs
        self.__dict__['_layer'] = Layer(numOfNeurons, numOfInputs, activFunc)

    def process(self, inputs):
        outputs = self._layer.processNeurons(inputs)
        outputsSum = np.sum(np.array(outputs))
        print(outputs)
        return self._thresholdFunc(outputsSum)

    class ThresholdFuncType(Enum):
        OR = 0
        AND = 1
        MAJORITY = 2

    def getThresholdFunc(self, numOfInputs, fType):
        if fType == Madaline.ThresholdFuncType.OR:
            return Sign()(1 - numOfInputs)
        elif fType == Madaline.ThresholdFuncType.AND:
            return Sign()(numOfInputs - 1)
        elif fType == Madaline.ThresholdFuncType.MAJORITY:
            return Sign()(0.0)

    def __getitem__(self, index):
        if index == 'layer':
            return self._layer
        elif index == 'thresholdFuncType':
            return self._thresholdFuncType


class LayerManager:
    def __init__(self, numOfLayers, numOfNeurons, numOfInputs, activFuncs):
        self.__dict__['_layers'] = []
        self.__dict__['_numOfLayers'] = numOfLayers
        self.__dict__['_numOfInputs'] = numOfInputs
        self.__dict__['_activFuncs'] = activFuncs

        for i in range(numOfLayers):
            self._layers.append(
                Layer(numOfNeurons[i], numOfInputs[i], activFuncs[i]))

    def processLayers(self, inputs):
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

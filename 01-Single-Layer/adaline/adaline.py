import random
import numpy as np
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
            if x > translation:
                return 1
            else:
                return -1
        return sign

class Adaline:
    def __init__(self, weights, activFunc, lRate = 0.05, bias=random.uniform(-1,1)):
        self.__dict__['_weights'] = weights
        self.__dict__['_bias'] = bias
        self.__dict__['_lRate'] = lRate
        self.__dict__['_activFunc'] = activFunc
        self.__dict__['_sum']=None
        self.__dict__['_val']=None

    def process(self, inputs):
        self._sum = np.dot(np.array(inputs), self._weights) + self._bias
        return self._activFunc(self._sum)

    def train(self, inputs, desired):
        guess = self.process(inputs)
        error = desired - self._sum

        for i in range(len(self._weights)):
            self._weights[i] += self._lRate * error * inputs[i]

        self._bias = self._lRate * error





class Layer:
    def __init__(self, numOfNeurons, numOfInputs, activFunc):
        self.__dict__['_neurons'] = []
        self.__dict__['_numOfNeurons'] = numOfNeurons
        self.__dict__['_activFunc'] = activFunc
        self.__dict__['_numOfInputs'] = numOfInputs

        for n in range(numOfNeurons):
            w = [random.uniform(-1,1) for _ in range(numOfInputs)]
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
        self.__dict__['_thresholdFunc'] = self.getThresholdFunc(numOfInputs, thresholdFuncType)
        self.__dict__['_numOfInputs'] = numOfInputs
        self.__dict__['_layer'] = Layer(numOfNeurons, numOfInputs, activFunc)



    def process(self, inputs):
        outputs =  self._layer.processNeurons(inputs)
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

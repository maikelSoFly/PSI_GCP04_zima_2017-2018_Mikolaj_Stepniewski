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
            return 1.0/(1.0+np.exp(-beta*x))
        sigm.__name__ += '({0:.3f})'.format(beta)
        return sigm
    def derivative(self, beta):
        def sigmDeriv(x):
            return beta*np.exp(-beta*x)/((1.0+np.exp(-beta*x))**2)
        sigmDeriv.__name__ += '({0:.3f})'.format(beta)
        return sigmDeriv

class SignSigm:
    def __call__(self,alfa):
        def signSigm(x):
            return (2.0/(1.0+np.exp(-alfa*x)))-1.0
        signSigm.__name__+='({0:.3f})'.format(alfa)
        return signSigm
    def derivative(self,alfa):
        def signSigmDeriv(x):
            return 2.0*alfa*np.exp(-alfa*x)/((1.0+np.exp(-alfa*x))**2)
        signSigmDeriv.__name__+='({0:.3f})'.format(alfa)
        return signSigmDeriv

def hardSign(x):
    if x<0:
        return -1.0
    return 1.0

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
    def __init__(self, numOfInputs, iid, activFunc, lRate=0.01, fRate=0.003, bias=-0.5):
        self._weights = np.array([random.uniform(-1, 1) for _ in range(numOfInputs)])
        self.__dict__['_activFunc'] = activFunc
        self.__dict__['_bias'] = bias
        self.__dict__['_lRate'] = lRate
        self.__dict__['_fRate'] = fRate     # forget rate
        self.__dict__['_trainingData'] = None
        self.__dict__['_error'] = None
        self.__dict__['_sum'] = None
        self.__dict__['_val'] = None
        self.__dict__['_iid'] = iid


    def process(self, inputs):
        self._sum = np.dot(self._weights, inputs) + self._bias
        self._val = self._activFunc(self._sum)
        return self._val

    def setTrainingData(self, data):
        for inputs in data:
            if len(inputs) != len(self._weights):
                raise Exception('Different number of weights and training set!')
        else:
            self._trainingData = data

    def train(self):
        if self._trainingData != None:
            for inputs in self._trainingData:
                output = self.process(inputs)

                self._error = output * self._lRate
                for i in range(len(self._weights)):
                    self._weights[i] *= 1.0 - self._fRate
                    self._weights[i] += self._lRate * output * inputs[i]
            self._bias *= (1-self._fRate)
            self._bias += output * self._lRate
        else:
            raise Exception('No training set.\n\tuse:\tsettrainingData(array)')


# class Layer:
#     def __init__(self, numOfNeurons, numOfInputs, activFunc, activFuncDeriv):
#         self.__dict__['_neurons'] = []
#         self.__dict__['_numOfNeurons'] = numOfNeurons
#         self.__dict__['_activFunc'] = activFunc
#         self.__dict__['_activFuncDeriv'] = activFuncDeriv
#         self.__dict__['_numOfInputs'] = numOfInputs
#
#         """ Creating neurons """
#         for n in range(numOfNeurons):
#             w = [random.uniform(-1, 1) for _ in range(numOfInputs)]
#             self._neurons.append(HebbNeuron(w, activFunc, activFuncDeriv))
#
#
#     def setTrainValues(self, inputs):
#         for neuron in self._neurons:
#             neuron.setTrainValues(inputs)
#
#     def processNeurons(self, inputs):
#         """ Passing data through the neurons of the layer.
#         Used for validation. """
#         outputs = []
#         for n in self._neurons:
#             outputs.append(n.process(inputs))
#         return outputs
#
#     def trainNeurons(self):
#         """ Passing training data through neurons of the layer. """
#         outputs = []
#         for index, n in enumerate(self._neurons):
#             if self._numOfNeurons > 1:
#                 n.train()
#             else:
#                 n.train()
#             outputs.append(n._val)
#         return outputs
#
#
# class LayerManager:
#     """ Class which manages single layer of neurons and Perceptron (processing
#     outputs of the layer). """
#     def __init__(self, numOfLayers, numOfNeurons, numOfInputs, activFuncs, activFuncDerivs):
#         self.__dict__['_layers'] = []
#         self.__dict__['_numOfLayers'] = numOfLayers
#         self.__dict__['_numOfInputs'] = numOfInputs
#         self.__dict__['_activFuncs'] = activFuncs
#         self.__dict__['_activFuncDerivs'] = activFuncDerivs
#
#         """ Creating single layers """
#         for i in range(numOfLayers):
#             self._layers.append(
#                 Layer(numOfNeurons[i], numOfInputs[i], activFuncs[i], activFuncDerivs[i]))
#
#     def processLayers(self, inputs):
#         """ Passing data through layers.
#         Used for validation. """
#         prevOuts = None
#         output = []
#         for i in range(self._numOfLayers):
#             if i == 0:
#                 prevOuts = self._layers[i].processNeurons(inputs)
#                 output.append(prevOuts)
#             else:
#                 prevOuts = self._layers[i].processNeurons(prevOuts)
#                 output.append(prevOuts)
#         return output
#
#     def trainLayers(self):
#         """ Passing training data through layers. """
#         for i in range(self._numOfLayers):
#             self._layers[i].trainNeurons()
#
#     def setTrainValues(self, inputs, iid):
#         self._layer[iid].setTrainValues(inputs)
#
#     """ Access method """
#     def __getitem__(self, index):
#         if index == 'layers':
#             return self._layers
#         elif index == 'numOfLayers':
#             return self._numOfLayers

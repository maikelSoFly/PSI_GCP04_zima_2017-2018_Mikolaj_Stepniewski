import numpy as np

def sign(x):
    if x >= 0:
        return 1
    else:
        return 0

def one(x):
    return 1

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

    def __init__(self, weights, activFunc, activFuncDeriv, lRate=0.5, bias=np.random.ranf()):
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
        self._error = delta * self._activFuncDeriv(self._sum)

        for i in range(len(self._weights)):
            self._weights[i] -= self._lRate * self._error * input[i]

        self._bias = self._lRate * self._error

    def __getitem__(self,index):
        if index=='val':
            return self._val

class Layer:
    def __init__(self, nNeurons, nInputs, activFunc, activFuncDeriv):
        self.__dict__['_neurons'] = []
        self.__dict__['_nNeurons'] = nNeurons
        self.__dict__['_activFunc'] = activFunc
        self.__dict__['_activFuncDeriv'] = activFuncDeriv
        self.__dict__['_outputs'] = []
        self.__dict__['_nIputs'] = nInputs

        for n in range(nNeurons):
            w = [np.random.ranf() for _ in range(nInputs)]
            self._neurons.append(Perceptron(w, activFunc, activFuncDeriv))

    def processOutputs(self, inputs):
        outputs = []
        for n in self._neurons:
            outputs.append(n.process(inputs))
        return outputs


    def trainNeurons(self, inputs, desired):
        outputs = []
        for n in self._neurons:
            n.train(inputs, desired)
            outputs.append(n._val)
        return outputs



class Multilayer:
    def __init__(self, nLayers, nNeurons, nInputs, activFuncs, activFuncDerivs):
        self.__dict__['_layers'] = []
        self.__dict__['_nLayers'] = nLayers
        self.__dict__['_nInputs'] = nInputs
        self.__dict__['_activFuncs'] = activFuncs
        self.__dict__['_activFuncDerivs'] = activFuncDerivs

        for i in range(nLayers):
            self._layers.append(Layer(nNeurons[i], nInputs[i], activFuncs[i], activFuncDerivs[i]))

    def processNetOutput(self, inputs):
        prevOuts = None
        for i in range(self._nLayers):
            if i == 0:
                prevOuts = self._layers[i].processOutputs(inputs)
                print('Layer1:', prevOuts)
            else:
                prevOuts = self._layers[i].processOutputs(prevOuts)
                print('Layer2:', prevOuts)

    def trainNet(self, inputVectors):
        prevOuts = None
        #print(inputVectors._x)

        for i in range(self._nLayers):
            if i == 0:
                prevOuts = self._layers[i].trainNeurons(inputVectors._x, inputVectors._d)
            else:
                prevOuts = self._layers[i].trainNeurons(prevOuts, inputVectors._d)

import sys
# Add the neuron folder path to the sys.path list
sys.path.append('../lib')
from neuron import *


class KohonenNeuron(Neuron):
    def __init__(self, numOfInputs, iid, lRate=0.01):
        Neuron.__init__(self, numOfInputs, iid, activFunc=None, lRate=lRate, bias=0)
        self.__dict__['_winnerCounter'] = 0
        self.__dict__['_pausedCounter'] = 0
        self.__dict__['_startWeights'] = self._weights


    def process(self, vector):
        self._sum = np.dot(self._weights, vector)
        return self._sum

    def train(self, vector):

        for i in range(len(self._weights)):
            self._weights[i] += self._lRate * (vector[i] - self._weights[i])

    def resetWeights(self):
        self._weights = self._startWeights


winnerLimit = 5
pauseFor = 3
class KohonenNeuronGroup:
    def __init__(self, numOfInputs, numOfNeurons, trainingData, lRate=0.01):
        self.__dict__['_numOfNeurons'] = numOfNeurons
        self.__dict__['_lRate'] = lRate
        self.__dict__['_numOfInputs'] = numOfInputs
        self.__dict__['_neurons'] = []
        self.__dict__['_trainingData'] = trainingData

        for i in range(numOfNeurons):
            neuron = KohonenNeuron(numOfInputs, i, lRate)
            neuron.setTrainingData(trainingData)
            self._neurons.append(neuron)

    def resetWeights(self):
        for neuron in self._neurons:
            neuron.resetWeights()

    def train(self, vectors):

        for vector in vectors:
            winner = None
            for neuron in self._neurons:
                neuron.process(vector)
                if winner == None:
                    winner = neuron
                elif winner != None:
                    if neuron._sum > winner._sum:
                        winner = neuron
                winner._winnerCounter += 1

        return winner

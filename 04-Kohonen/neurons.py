import sys
# Add the neuron folder path to the sys.path list
sys.path.append('../lib')
from neuron import *


class KohonenNeuron(Neuron):
    def __init__(self, numOfInputs, iid, lRate=0.01):
        Neuron.__init__(self, numOfInputs, iid, activFunc=None, lRate=lRate, bias=0)
        self.__dict__['_winnerCounter'] = 0
        self.__dict__['_pausedCounter'] = 0


    def process(self, vector):
        self._sum = np.dot(self._weights, vector)

        return self._sum

    def train(self, vector):

        for i in range(len(self._weights)):
            self._weights[i] += self._lRate * (vector[i] - self._weights[i])


winnerLimit = 10000
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

    def train(self):

        for vector in self._trainingData:
            winner = None
            for neuron in self._neurons:
                u = neuron.process(vector)
                if winner == None:
                    winner = neuron
                else:
                    if u > winner._sum and neuron._winnerCounter <= winnerLimit:
                        winner = neuron
                winner._winnerCounter += 1

            print('Winner is:\t', winner._iid, 'with u:\t', winner._sum)
            winner.train(vector)

        for neuron in self._neurons:
            neuron._winnerCounter = 0

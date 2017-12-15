import sys
# Add the neuron folder path to the sys.path list
sys.path.append('../inc')
from neuron import *
from collections import Counter


class KohonenNeuron(Neuron):
    def __init__(self, numOfInputs, processFunc, iid, lRate=0.1):
        Neuron.__init__(self, numOfInputs, iid, activFunc=None, lRate=lRate, bias=0)
        self.__dict__['_winnerCounter'] = 0
        self.__dict__['_pausedCounter'] = 0
        self.__dict__['_processFunc'] = processFunc
        self.__dict__['_startWeights'] = self._weights[:]

    def process(self, vector):
        self._sum = self._processFunc(vector, self._weights)
        return self._sum

    def train(self, vector):
        for i in range(len(self._weights)):
            self._weights[i] += self._lRate * (vector[i] - self._weights[i])

    def resetWeights(self):
        self._weights = self._startWeights[:]


""" Simple WTA for now...

    Winner is neuron with least distance
    between weights vector and input vector """
class KohonenNeuronGroup:
    def __init__(self, numOfInputs, numOfNeurons, processFunc, trainingData, lRateFunc, lRate=0.1):
        self.__dict__['_numOfNeurons'] = numOfNeurons
        self.__dict__['_lRate'] = lRate
        self.__dict__['_numOfInputs'] = numOfInputs
        self.__dict__['_neurons'] = None
        self.__dict__['_trainingData'] = trainingData
        self.__dict__['_processFunc'] = processFunc
        self.__dict__['_lRateFunc'] = lRateFunc
        self.__dict__['_currentLRate'] = None

        self._neurons = [[KohonenNeuron(numOfInputs, processFunc, iid=i*numOfNeurons[0]+j, lRate=lRate)
            for i in range(numOfNeurons[0])]
            for j in range(numOfNeurons[1])
        ]


    def resetWeights(self):
        for row in self._neurons:
            for neuron in row:
                neuron.resetWeights()

    def resetWins(self):
        for row in self._neurons:
            for neuron in row:
                neuron._winnerCounter = 0


    def train(self, vectors, retMostCommon=False):
        winner = None
        winners = []
        for i, vector in enumerate(vectors):
            winner = None
            for row in self._neurons:
                for neuron in row:
                    neuron.process(vector)
                    if winner == None:
                        winner = neuron
                    elif winner != None:
                        if neuron._sum < winner._sum:
                            winner = neuron

                    winner._winnerCounter += 1

            winners.append(winner)

        self._currentLRate = self._lRate * self._lRateFunc()

        if retMostCommon:
            return Counter(winners).most_common(1)[0][0]

        return np.split(np.array(winners), 3)


    """ Access methods """
    def __getitem__(self, key):
        if key == 'totalNumOfNeurons':
            return sum(len(x) for x in self._neurons)

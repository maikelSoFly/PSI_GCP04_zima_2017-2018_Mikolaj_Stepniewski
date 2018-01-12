import sys
# Add the include folder path to the sys.path list
sys.path.append('../include')
from supportFunctions import *
from collections import Counter


class HebbNeuron:
    def __init__(self, numOfInputs, iid, activFunc, lRate=0.1, fRate=0.28, bias=-0.5):
        self._weights = np.array([np.random.uniform(-1, 1) for _ in range(numOfInputs)])
        self.__dict__['_bias'] = bias
        self.__dict__['_activFunc'] = activFunc
        self.__dict__['_lRate'] = lRate
        self.__dict__['_fRate'] = fRate     # forget rate
        self.__dict__['_trainingData'] = None
        self.__dict__['_sum'] = None
        self.__dict__['_val'] = None
        self.__dict__['_iid'] = iid
        self.__dict__['_sumHist'] = []
        self.__dict__['_winnerCounter'] = 0

    def process(self, inputs):
        """ Compute y_i (simple dot) """
        self._sum = np.dot(self._weights, inputs) + self._bias
        self._val = self._activFunc(self._sum)
        return self._val


    def train(self, inputs):
        """ Get y_i """
        output = self.process(inputs)
        constant = self._lRate * output
        forget = (1.0 - self._fRate)
        for i in range(len(self._weights)):
            """ Apply forgetting """
            self._weights[i] *= forget
            """ Update weights """
            self._weights[i] += constant * inputs[i]

        # self._bias *= 1.0 - self._fRate
        # self._bias += constant



""" Simple WTA for now...

    Winner is neuron with least distance
    between weights vector and input vector """
class HebbNeuronGroup:
    def __init__(self, numOfInputs, numOfNeurons, activFunc, lRateFunc, lRate=0.007, fRate=0.1):
        self.__dict__['_numOfNeurons'] = numOfNeurons
        self.__dict__['_activFunc'] = activFunc
        self.__dict__['_lRate'] = lRate
        self.__dict__['_fRate'] = fRate
        self.__dict__['_numOfInputs'] = numOfInputs
        self.__dict__['_neurons'] = None
        self.__dict__['_lRateFunc'] = lRateFunc
        self.__dict__['_currentLRate'] = None

        """ Create group of neurons """
        self._neurons = [[HebbNeuron(numOfInputs, i*numOfNeurons[0]+j, activFunc, lRate=lRate, fRate=fRate)
            for i in range(numOfNeurons[0])]
            for j in range(numOfNeurons[1])
        ]

    """ Reselting weights of all neurons in group """
    def resetWeights(self):
        for row in self._neurons:
            for neuron in row:
                neuron.resetWeights()

    def resetWins(self):
        for row in self._neurons:
            for neuron in row:
                neuron._winnerCounter = 0

    def setLRate(self, lRate):
        self._currentLRate = lRate
        for row in self._neurons:
            for neuron in row:
                neuron._lRate = lRate


    def train(self, vectors, histFreq=1, retMostCommon=False):
        winners = []
        """ Finding winner (highest value) """
        for i, vector in enumerate(vectors):
            winner = None
            for row in self._neurons:
                for neuron in row:
                    neuron.process(vector)
                    if winner == None:
                        winner = neuron
                    elif winner != None:
                        if neuron._val > winner._val:
                            winner = neuron

            """ Winner Takes All """
            if winner._winnerCounter % histFreq == 0:
                winner._sumHist.append(winner._sum)
            winner._winnerCounter += 1
            winner.train(vector)
            winners.append(winner)

        """ Updating lRate """
        self.setLRate(self._lRateFunc(self._lRate))

        if retMostCommon:
            return Counter(winners).most_common(1)[0][0]

        return winners

    """ Basicaly the same as above, but without updating weights """
    def classify(self, vectors):
        winners = []
        for i, vector in enumerate(vectors):
            winner = None
            for row in self._neurons:
                for neuron in row:
                    neuron.process(vector)
                    if winner == None:
                        winner = neuron
                    elif winner != None:
                        if neuron._val > winner._val:
                            winner = neuron

            winners.append(winner)

        return winners


    """ Access methods """
    def __getitem__(self, key):
        if key == 'totalNumOfNeurons':
            return sum(len(x) for x in self._neurons)

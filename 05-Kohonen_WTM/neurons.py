# @Author: Mikołaj Stępniewski <maikelSoFly>
# @Date:   2017-12-12T17:15:51+01:00
# @Email:  mikolaj.stepniewski1@gmail.com
# @Filename: neurons.py
# @Last modified by:   maikelSoFly
# @Last modified time: 2017-12-16T23:31:39+01:00
# @License: Apache License  Version 2.0, January 2004
# @Copyright: Copyright © 2017 Mikołaj Stępniewski. All rights reserved.



import sys
# Add the include folder path to the sys.path list
sys.path.append('../include')
from neuron import *
from supportFunctions import *
from collections import Counter
import copy


class KohonenNeuron(Neuron):
    def __init__(self, x, y, numOfInputs, processFunc, iid, lRate=0.1):
        Neuron.__init__(self, numOfInputs, iid, activFunc=None, lRate=lRate, bias=0)
        self.__dict__['_winnerCounter'] = 0
        self.__dict__['_processFunc'] = processFunc
        self.__dict__['_startWeights'] = copy.deepcopy(self._weights)
        self.__dict__['_errorHist'] = []
        self.__dict__['_iid'] = iid
        self.__dict__['_x'] = x
        self.__dict__['_y'] = y


    def process(self, vector):
        """ Euklides distance between input vector and weights vector """
        self._error = self._processFunc(vector, self._weights)
        return self._error


    def train(self, vector, G):
        """ WTM train with magnitude based on neighbourhood function """
        const = self._lRate * G
        for i in range(len(self._weights)):
            self._weights[i] +=  const * (vector[i] - self._weights[i])


    def resetWeights(self):
        self._weights = self._startWeights[:]




""" Simple WTA for now...

    Winner is neuron with least distance
    between weights vector and input vector """
class KohonenNeuronGroup:
    def __init__(self, numOfInputs, numOfNeurons, processFunc, lRateFunc, neighbourhoodRadius, nRadiusFunc, lRate=0.1):
        self.__dict__['_numOfNeurons'] = numOfNeurons
        self.__dict__['_lRate'] = lRate
        self.__dict__['_numOfInputs'] = numOfInputs
        self.__dict__['_neurons'] = None
        self.__dict__['_processFunc'] = processFunc
        self.__dict__['_lRateFunc'] = lRateFunc
        self.__dict__['_currentLRate'] = None
        self.__dict__['_neighbourhoodRadius'] = neighbourhoodRadius
        self.__dict__['_neighbourhoodRadiusFunc'] = nRadiusFunc

        self._neurons = [[KohonenNeuron(j, i, numOfInputs, processFunc, iid=i*numOfNeurons[0]+j, lRate=lRate)
            for i in range(numOfNeurons[0])]
            for j in range(numOfNeurons[1])
        ]

    """ Reset training parameters and reset weights to initial ones """
    def resetGroup(self, lRate, radius, radiusFunc):
        self._lRate = lRate
        self._neighbourhoodRadius = radius
        self._neighbourhoodRadiusFunc = radiusFunc
        for row in self._neurons:
            for neuron in row:
                neuron._weights = copy.deepcopy(neuron._startWeights)


    def resetWins(self):
        for row in self._neurons:
            for neuron in row:
                neuron._winnerCounter = 0


    def setLRate(self, lRate):
        self._currentLRate = lRate
        for row in self._neurons:
            for neuron in row:
                neuron._lRate = lRate


    def setNeighbourhoodRadius(self):
        self._neighbourhoodRadius = self._neighbourhoodRadiusFunc(self._neighbourhoodRadius)


    def trainNeighbours(self, vector, winner):
        coordsW = (winner._x, winner._y)
        for row in self._neurons:
            for neuron in row:
                neuron.train(vector, self.gaussNeighbourhood(coordsW, [neuron._x, neuron._y]))


    def gaussNeighbourhood(self, coordsW, coordsI):
        dx = (coordsW[0]-coordsI[0])
        dy = (coordsW[1]-coordsI[1])
        dist = dx*dx + dy*dy
        return np.exp(-dist/(2*self._neighbourhoodRadius**2))


    def train(self, vector):
        winner = None
        for row in self._neurons:
            for neuron in row:
                neuron.process(vector)
                if winner == None:
                    winner = neuron
                elif winner != None:
                    if neuron._error < winner._error:
                        winner = neuron

        """ Winner Takes MOST """
        """ Updating weights with magnitude based on neighbourhood func """
        winner.train(vector, 1)
        self.trainNeighbours(vector, winner)
        self.setLRate(self._lRateFunc(self._lRate))

        return winner


    """ Basicaly the same as above, but without updating weights """
    def classify(self, vector):
        winner = None
        for row in self._neurons:
            for neuron in row:
                neuron.process(vector)
                if winner == None:
                    winner = neuron
                elif winner != None:
                    if neuron._error < winner._error:
                        winner = neuron

        return winner


    """ Access methods """
    def __getitem__(self, key):
        if key == 'totalNumOfNeurons':
            return sum(len(x) for x in self._neurons)

# @Author: Mikołaj Stępniewski <maikelSoFly>
# @Date:   2017-12-16T02:09:12+01:00
# @Email:  mikolaj.stepniewski1@gmail.com
# @Filename: main.py
# @Last modified by:   maikelSoFly
# @Last modified time: 2017-12-17T15:20:01+01:00
# @License: Apache License  Version 2.0, January 2004
# @Copyright: Copyright © 2017 Mikołaj Stępniewski. All rights reserved.



from math import ceil
from math import floor
from neurons import *
from data import *
from progressBar import *
import random


dataUrl = 'http://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
speciesNames = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']

""" Training parameters """
epochs = 10
decay = (epochs)*150
neuronGrid = [17, 17]
lRate = 0.09    # 0.07 one of the best


def countUniqueItems(arr):
    return len(Counter(arr).keys())

def getMostCommonItem(arr):
    return Counter(arr).most_common(1)[0][0]

def averageParameters(species, n=50):
    sum = [0.0 for _ in range(4)]
    for row in species:
        sum[0] += row[0]
        sum[1] += row[1]
        sum[2] += row[2]
        sum[3] += row[3]
    return [ceil((sum[i]/n)*100)/100 for i in range(4)]


def trainSeparately(kohonenGroup, speciesArr):
    pBar = ProgressBar()
    winners = []
    for j, species in enumerate(speciesArr):
        print('\n', speciesNames[j])
        pBar.start(maxVal=epochs)
        for i in range(epochs):
            pBar.update()
            """ Train with one species at a time """
            winner = kohonenGroup.train(species, retMostCommon=True)
        winners.append(winner)  # winner for each species
        kohonenGroup.resetWeights()
        """ ^ reset weights between species """

    return winners

""" Main training function !!! """
def train(kohonenGroup, trainingData):
    pBar = ProgressBar()
    print('\n {} + {} + {}'.format(speciesNames[0], speciesNames[1], speciesNames[2]))
    pBar.start(maxVal=epochs)

    for i in range(epochs):
        winners = kohonenGroup.train(trainingData, histFreq=20)
        pBar.update()

    return winners




trainingData = DataReader(url=dataUrl, delimiter=',').parse()

for j in range(len(trainingData)):
        trainingData[j].pop()                                   # remove species name
        trainingData[j] = [float(i) for i in trainingData[j]]   # cast str elements to float
        trainingData[j] = normalizeInputs(trainingData[j])      # normalize elements to 0...1 values

speciesArr = np.split(np.array(trainingData), 3)                # split in 3 different species arrays


kohonenGroup = KohonenNeuronGroup(
    numOfInputs=4,
    numOfNeurons=neuronGrid,
    processFunc=euklidesDistance,
    trainingData=trainingData,
    lRateFunc=simpleLRateCorrection(decay),
    lRate=lRate
)


print('lRate0: {:.2f}\tdecay: {}\tneurons in group: {:d}\tepochs: {:d}'.format(
    kohonenGroup._lRate, decay, kohonenGroup['totalNumOfNeurons'], epochs
))

print('\n•Averages:')
for i, species in enumerate(speciesArr):
    print('{} \t{}'.format(averageParameters(species), speciesNames[i]))
print()




""" Training & results """

# winners = trainSeparately(kohonenGroup, speciesArr)
# print('\n\n•Results:\t(Most common winner-neurons)')
# for i, winner in enumerate(winners):
#     print('idd: {} \t{}\t{}'.format(winner._iid, winner._weights, speciesNames[i]))
#
# print('\n')

#random.shuffle(trainingData)

winners = train(kohonenGroup, trainingData)
numOfActiveNeurons = countUniqueItems(winners)
winners = np.split(np.array(winners), 3)

print('\n\n•Results:')
for i, row in enumerate(winners):
    print(' {}:\n   active neurons: {:d}\n   most common neuron: {:d}\n'.format(speciesNames[i], countUniqueItems(row), getMostCommonItem(row)._iid))
    for j, n in enumerate(row):
        if j != 0 and j % 10 == 0:
            print()
        print(n._iid, end=' ', flush=False)
    print('\n')

mostActiveNeurons = [getMostCommonItem(row) for row in winners]

print('\n•Most common active neurons for species:')
for i, neuron in enumerate(mostActiveNeurons):
    print('idd: {}  \t{}\t{}'.format(neuron._iid, neuron._weights, speciesNames[i]))

print('\n•Total active neurons in group: {:d}'.format(numOfActiveNeurons))
print('\nlRate{:d}: {:.5f}'.format(epochs, kohonenGroup._currentLRate))

answ = input('Print error history?\ty/n: ')
if answ == 'y':
    for neuron in mostActiveNeurons:
        print('\n')
        print('▄' * 25, '   [neuron: {:d}]\n\n'.format(neuron._iid))
        for row in neuron._errorHist:
            print(row)

import sys
import random
from math import ceil
from math import floor
# Add the activFuncs folder path to the sys.path list
sys.path.append('../inc')
from supportFunctions import *
from neurons import *
from data import *
import time

dataUrl = 'http://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
speciesNames = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']

""" Training parameters """
epochs = 10
lRateLambda = 1*149
neuronGrid = [17, 17]
lRate = 0.1


def countUniqueItems(arr):
    return len(Counter(arr).keys())

def getMostCommonItem(arr):
    return Counter(arr).most_common(1)[0][0]

def averageParameters(species, n=50):
    sum = [0.0, 0.0, 0.0, 0.0]
    for row in species:
        sum[0] += row[0]
        sum[1] += row[1]
        sum[2] += row[2]
        sum[3] += row[3]
    return [ceil((sum[0]/n)*100)/100, ceil((sum[1]/n)*100)/100, ceil((sum[2]/n)*100)/100, ceil((sum[3]/n)*100)/100]


def trainSeparately(kohonenGroup, speciesArr):
    winners = []
    start = time.time()
    for j, species in enumerate(speciesArr):
        print('\n', speciesNames[j])
        print('....................')
        for i in range(epochs):
            if i != 0 and i % (round(epochs/10)) == 0:
                print('▇', end=' ', flush=True)
            """ Train with one species at a time """
            winner = kohonenGroup.train(species, retMostCommon=True)
        winners.append(winner)  # winner for each species
        kohonenGroup.resetWeights()
        """ ^ reset weights between species """
        end = time.time()
        print('▇\tdone\tin: {:.3f} sec'.format(end-start))

    return winners

def trainSimultaneously(kohonenGroup, trainingData):
    print('\n {} + {} + {}'.format(speciesNames[0], speciesNames[1], speciesNames[2]))
    print('....................')
    start = time.time()
    for i in range(epochs):
        if i != 0 and i % (round(epochs/10)) == 0:
            print('▇', end=' ', flush=True)
        """ Train with one species at a time """
        winners = kohonenGroup.train(trainingData)
    end = time.time()
    print('▇\tdone\tin: {:.3f} sec'.format(end-start))
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
    lRateFunc=simpleLRateCorrection(lRateLambda),
    lRate=lRate
)


print('lRate0: {:.2f}\tlRateLambda: {}\tneurons in group: {:d}\tepochs: {:d}'.format(
    kohonenGroup._lRate, lRateLambda, kohonenGroup['totalNumOfNeurons'], epochs
))

print('\n•Averages:')
for i, species in enumerate(speciesArr):
    print('{} \t{}'.format(averageParameters(species), speciesNames[i]))
print()




""" Training """

# winners = trainSeparately(kohonenGroup, speciesArr)
# print('\n\n•Results:\t(Most common winner-neurons)')
# for i, winner in enumerate(winners):
#     print('idd: {} \t{}\t{}'.format(winner._iid, winner._weights, speciesNames[i]))
#
# print('\n')

winners = trainSimultaneously(kohonenGroup, trainingData)
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
print('\n•lRate: {:.5f}'.format(kohonenGroup._currentLRate))

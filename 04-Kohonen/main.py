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

epochs = 100
lRateLambda = 100*150

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
            if i != 0 and i % (floor(epochs/10)) == 0:
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
        if i != 0 and i % (floor(epochs/10)) == 0:
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
    numOfNeurons=[17, 17],
    processFunc=euklidesDistance,
    trainingData=trainingData,
    lRateFunc=simpleLRateCorrection(lRateLambda),
    lRate=0.1
)


print('lRate0: {:.2f}\tlRateLambda: {}\tneurons in group: {:d}\tepochs: {:d}'.format(
    kohonenGroup._lRate, lRateLambda, kohonenGroup['totalNumOfNeurons'], epochs
))

print('\n•Averages:')
for i, species in enumerate(speciesArr):
    print('{} \t{}'.format(averageParameters(species), speciesNames[i]))
print()




""" Training """

winners = trainSeparately(kohonenGroup, speciesArr)
print('\n\n•Results:\t(Most common winner-neurons)')
for i, winner in enumerate(winners):
    print('idd: {} \t{}\t{}'.format(winner._iid, winner._weights, speciesNames[i]))

print('\n')

winners = trainSimultaneously(kohonenGroup, trainingData)
print('\n\n•Results:')
for i, row in enumerate(winners):
    print(' {}:'.format(speciesNames[i]))
    for j, n in enumerate(row):
        if j != 0 and j % 10 == 0:
            print()
        print(n._iid, end=' ', flush=False)
    print('\n')

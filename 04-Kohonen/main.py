import sys
import random
from math import ceil
# Add the activFuncs folder path to the sys.path list
sys.path.append('../lib')
from supportFunctions import *
from neurons import *
from data import *

dataUrl = 'http://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
speciesNames = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']

epochs = 100

def averageParameters(species, n=50):
    sum = [0.0, 0.0, 0.0, 0.0]
    for row in species:
        sum[0] += row[0]
        sum[1] += row[1]
        sum[2] += row[2]
        sum[3] += row[3]
    return [ceil((sum[0]/n)*100)/100, ceil((sum[1]/n)*100)/100, ceil((sum[2]/n)*100)/100, ceil((sum[3]/n)*100)/100]

def normalizeInputs(arr):
    sum = 0.0
    for el in arr:
        sum += el**2
    return [i/sqrt(sum) for i in arr]



trainingData = DataReader(url=dataUrl, delimiter=',').parse()

for j in range(len(trainingData)):
        trainingData[j].pop()                                   # remove species name
        trainingData[j] = [float(i) for i in trainingData[j]]   # cast str elements to float
        trainingData[j] = normalizeInputs(trainingData[j])      # normalize elements to 0...1 values

speciesArr = np.split(np.array(trainingData), 3)                # split in 3 different species arrays


kohonenGroup = KohonenNeuronGroup(
    numOfInputs=4,
    numOfNeurons=225,
    processFunc=euklidesDistance,
    trainingData=trainingData,
    lRateFunc=simpleLearnCorrection(20*50),
    lRate=0.1
)


print('lRate: {:.2f}\tneurons in group: {:d}\tepochs: {:d}'.format(
    kohonenGroup._lRate, kohonenGroup._numOfNeurons, epochs
))

print('\n•Averages:')
for i, species in enumerate(speciesArr):
    print('{} \t{}'.format(averageParameters(species), speciesNames[i]))
print()

winners = []
for j, species in enumerate(speciesArr):
    print('\n', speciesNames[j])
    print('....................')
    for i in range(epochs):
        if i != 0 and i % (epochs/10) == 0:
            print('▇', end=' ', flush=True)
        """ Train with one species at a time """
        winner = kohonenGroup.train(species)
    winners.append(winner)  # winner for each species
    kohonenGroup.resetWeights()
    """ ^ reset weights between species """
    print('▇\tdone')


print('\n\n•Results:')
for i, winner in enumerate(winners):
    print('idd: {} \t{}\t{}'.format(winner._iid, winner._weights, speciesNames[i]))

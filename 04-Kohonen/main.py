import sys
import random
# Add the activFuncs folder path to the sys.path list
sys.path.append('../lib')
from activFuncs import *
from neurons import *
from data import *

dataUrl = 'http://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
speciesNames = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']


trainingData = DataReader(url=dataUrl, delimiter=',').parse()



for i in range(len(trainingData)):
        trainingData[i].pop()
        trainingData[i] = [float(i) for i in trainingData[i]]

speciesArr = np.split(np.array(trainingData), 3)


kohonenGroup = KohonenNeuronGroup(numOfInputs=4, numOfNeurons=35, trainingData=trainingData, lRate=0.007)

winners = []
for j, species in enumerate(speciesArr):
    for i in range(1000):
        winner = kohonenGroup.train(species)
    winners.append(winner)
    print(speciesNames[j], ' finished...')
    kohonenGroup.resetWeights()

for i, winner in enumerate(winners):
    print('idd: ', winner._iid, winner._weights, '\t', speciesNames[i])

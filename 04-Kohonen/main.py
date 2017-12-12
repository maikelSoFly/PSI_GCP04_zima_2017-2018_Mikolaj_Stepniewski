import sys
import random
# Add the activFuncs folder path to the sys.path list
sys.path.append('../lib')
from activFuncs import *
from neurons import *
from data import *

dataUrl = 'http://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'


trainingData = DataReader(url=dataUrl, delimiter=',').parse()
print('Len: ', len(trainingData))
for i in range(len(trainingData)):
    trainingData[i].pop()
    trainingData[i] = [float(i) for i in trainingData[i]]

speciesArr = np.split(np.array(trainingData), 3)
print(len(speciesArr))





kohonenGroup = KohonenNeuronGroup(numOfInputs=4, numOfNeurons=35, trainingData=trainingData, lRate=0.001)

winners = []
for species in speciesArr:
    for i in range(100):
        winner = kohonenGroup.train(species)
    winners.append(winner)

for winner in winners:
    print(winner._weights)

from neurons import *
import numpy as np


trainingSet = [1,0,1,0,0,0,0,1]

hebbNr = HebbNeuron(numOfInputs=8, iid=0, activFunc=Sigm()(1.0), lRate=0.01, fRate=0.01/3)

hebbNr.setTrainValues(trainingSet)

for i in range(300):
    hebbNr.train()

print('Hebb successfully trained!')

from neurons import *
import numpy as np
from emojis import *

emoji = Emoji()

trainingSet = [emoji.getEmoji('xD'), emoji.getEmoji('sad'), emoji.getEmoji('angry'),
emoji.getEmoji('confused'), emoji.getEmoji('smile')]

hebbNr = HebbNeuron(numOfInputs=64, iid=0, activFunc=SignSigm()(1.0), lRate=0.01, fRate=0.003)

hebbNr.setTrainValues(trainingSet)

for i in range(1000):
    hebbNr.train()

for emoji in trainingSet:
    print(hebbNr.process(emoji))

print('Hebb successfully trained!')

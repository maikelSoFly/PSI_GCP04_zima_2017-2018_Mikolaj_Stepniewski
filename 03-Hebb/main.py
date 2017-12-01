from neurons import *
import numpy as np
from emojis import *

emoji = Emoji()

trainingSet = [
    emoji.getEmoji('xD'),
    emoji.getEmoji('sad'),
    emoji.getEmoji('angry'),
    emoji.getEmoji('confused'),
    emoji.getEmoji('smile'),
    emoji.getEmoji('test')
]

hebbNr = HebbNeuron(
    numOfInputs=64,
    iid=0,
    activFunc=SignSigm()(1.0),
    lRate=0.007,
    fRate=0.1
)

hebbNr.setTrainingData(trainingSet)

for i in range(10000):
    hebbNr.train()

for emoji in trainingSet:
    print(hebbNr.process(emoji))

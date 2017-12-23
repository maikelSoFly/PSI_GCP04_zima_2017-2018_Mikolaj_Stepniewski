from neurons import *
import numpy as np
from emojis import *
from random import randint

#np.random.seed(5)
neuronGrid = [15, 15]
lRate=0.007
fRate=0.4


emoji = Emoji()

def countUniqueItems(arr):
    return len(Counter(arr).keys())

def noiseEmojis(arr, numOfPixels):
    noisedArr = arr[:]
    for emoji in noisedArr:
        for _ in range(numOfPixels):
            emoji[randint(0, 63)] *= -1

    return noisedArr


trainingSet = [
    emoji.getEmoji('sad'),
    emoji.getEmoji('smile'),
    emoji.getEmoji('angry'),
    emoji.getEmoji('xD'),
    emoji.getEmoji('confused'),
    emoji.getEmoji('test')
]

noisedSet = noiseEmojis(trainingSet, 3)


hebbGroup = HebbNeuronGroup(
    numOfInputs=64,
    numOfNeurons=neuronGrid,
    processFunc=SignSigm()(0.5),
    lRateFunc=Linear()(),
    lRate=lRate,
    fRate=fRate
)

winners = []
for i in range(100):
    winners = hebbGroup.train(trainingSet+noisedSet)

for winner in winners:
    print(winner._iid)

print('Active neurons: {:d}'.format(countUniqueItems(winners)))

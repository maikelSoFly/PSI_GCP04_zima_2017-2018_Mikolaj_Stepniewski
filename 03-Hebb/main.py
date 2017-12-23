from neurons import *
import numpy as np
from emojis import *
from random import randint
import copy

#np.random.seed(5)
neuronGrid = [15, 15]
lRate=0.1
fRate=0.28


def countUniqueItems(arr):
    return len(Counter(arr).keys())

def noiseEmojis(arr, numOfPixels):
    noisedArr = copy.deepcopy(arr)
    for emoji in noisedArr:
        pixels = np.random.choice(64, numOfPixels, replace=False)
        for pixel in pixels:
            emoji[pixel] *= -1

    return noisedArr



emoji = Emoji()
trainingSet = [
    emoji.getEmoji('sad'),
    emoji.getEmoji('smile'),
    emoji.getEmoji('angry'),
    emoji.getEmoji('xD'),
    emoji.getEmoji('confused'),
    emoji.getEmoji('test')
]
noisedSet = noiseEmojis(trainingSet, 4)


hebbGroup = HebbNeuronGroup(
    numOfInputs=64,
    numOfNeurons=neuronGrid,
    activFunc=SignSigm()(0.5),
    lRateFunc=Linear()(),
    lRate=lRate,
    fRate=fRate
)


for i in range(50):
    """ Will get winners from the last epoch """
    winners = hebbGroup.train(trainingSet+noisedSet)

numOfActiveNeurons = countUniqueItems(winners)
winners = np.split(np.array(winners), 2)

for i in range(len(winners[0])):
    print(winners[0][i]._iid, winners[1][i]._iid)

print('Active neurons: {:d}'.format(numOfActiveNeurons))

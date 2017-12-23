from neurons import *
import numpy as np
from emojis import *
from random import randint
import random
import copy

#np.random.seed(5)
neuronGrid = [15, 15]
lRate=0.007
fRate=0.4


emoji = Emoji()

def countUniqueItems(arr):
    return len(Counter(arr).keys())

def noiseEmojis(arr, numOfPixels):
    noisedArr = copy.deepcopy(arr)
    for emoji in noisedArr:
        pixels = random.sample(range(1, 63), numOfPixels)
        for pixel in pixels:
            emoji[pixel] *= -1

    return noisedArr



trainingSet = [
    emoji.getEmoji('sad'),
    emoji.getEmoji('smile'),
    emoji.getEmoji('angry'),
    emoji.getEmoji('xD'),
    emoji.getEmoji('confused'),
    emoji.getEmoji('test')
]

noisedSet = noiseEmojis(trainingSet, 2)
# for emoji in trainingSet+noisedSet:
#     for i in range(64):
#         if i != 0 and i % 8 == 0:
#             print()
#         print(emoji[i], end=' ', flush=True)
#     print('\n')

trainingSet += noisedSet


hebbGroup = HebbNeuronGroup(
    numOfInputs=64,
    numOfNeurons=neuronGrid,
    processFunc=SignSigm()(0.5),
    lRateFunc=Linear()(),
    lRate=lRate,
    fRate=fRate
)

# for neuron in hebbNrs:
#     neuron.setTrainingData(trainingSet)

winners = []
for i in range(20):
    winners = hebbGroup.train(trainingSet)

winners = np.split(np.array(winners), 2)

for i in range(len(winners[0])):
    print(winners[0][i]._iid, winners[1][i]._iid)

print('Active neurons: {:d}'.format(countUniqueItems(np.concatenate((winners[0],winners[1]), axis=0))))

from neurons import *
from emojis import *
import copy

#np.random.seed(5)
neuronGrid = [15, 15]
lRate=0.007
fRate=0.35


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
""" Working well up to 6 noise pixels """
noisedSet = noiseEmojis(trainingSet, 6)


hebbGroup = HebbNeuronGroup(
    numOfInputs=64,
    numOfNeurons=neuronGrid,
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

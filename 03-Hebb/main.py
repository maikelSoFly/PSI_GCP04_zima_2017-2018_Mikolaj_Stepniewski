from neurons import *
from emojis import *
import copy

#np.random.seed(5)
neuronGrid = [10, 10]
lRate=0.007
fRate=0.36
numOfNoisePixels=5

def bipolarEmoji(emoji):
    for i in range(len(emoji)):
        if emoji[i] == 0:
            emoji[i] = -1
    return emoji

def countUniqueItems(arr):
    return len(Counter(arr).keys())

def noiseEmojis(arr, numOfNoisePixels):
    noisedArr = copy.deepcopy(arr)
    for emoji in noisedArr:
        pixels = np.random.choice(64, numOfNoisePixels, replace=False)
        for pixel in pixels:
            emoji[pixel] *= -1

    return noisedArr



emoji = Emoji()
emojisToGet = [ 'sad', 'smile', 'angry', 'laugh', 'surprised', 'confused' ]
trainingSet = [ bipolarEmoji(emoji.getEmoji(name)) for name in emojisToGet ]

""" Working well up to 6 noise pixels """
noisedSet = noiseEmojis(trainingSet, numOfNoisePixels)


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

print('NORMAL\tNOISED')
for i in range(len(winners[0])):
    print(winners[0][i]._iid, '\t', winners[1][i]._iid)

print('Active neurons: {:d}'.format(numOfActiveNeurons))

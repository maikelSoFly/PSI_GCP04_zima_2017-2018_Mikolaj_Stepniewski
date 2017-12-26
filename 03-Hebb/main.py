from neurons import *
from emojis import *
import copy

#np.random.seed(5)
neuronGrid = [10, 10]
lRate=0.01
fRate=0.02
numOfNoisePixels=3
epochs=50

def bipolar(emoji):
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

def drawEmojis(emojis):
    i = 0
    emojis = np.split(np.array(emojis), 2)
    for j in range(len(emojis)):
        for row in range(8):
            for emoji in emojis[j]:
                for i in range(8):
                    print('◼️' if emoji[row*8+i] == -1 or emoji[row*8+i] == 0 else '◻️', end=' ', flush=True)
                    if (i+1) % 8 == 0:
                        print('    ', end='', flush=False)
                        pass
            print()
        print('\n')



emoji = Emoji()
emojisToGet = [ 'sad', 'smile', 'angry', 'laugh', 'surprised', 'confused' ]
trainingSet = [ bipolar(emoji.getEmoji(name)) for name in emojisToGet ]

""" Working well up to 5 noise pixels """
noisedSet = noiseEmojis(trainingSet, numOfNoisePixels)

drawEmojis(trainingSet)
print('NOISED:\n')
drawEmojis(noisedSet)

trainingSet = normalizeInputs2d(trainingSet)
noisedSet = normalizeInputs2d(noisedSet)





hebbGroup = HebbNeuronGroup(
    numOfInputs=64,
    numOfNeurons=neuronGrid,
    activFunc=SignSigm()(1.0),
    lRateFunc=Linear()(),
    lRate=lRate,
    fRate=fRate
)


for i in range(epochs):
    hebbGroup.train(trainingSet)


winners = hebbGroup.classify(trainingSet+noisedSet)

numOfActiveNeurons = countUniqueItems(winners)
winners = np.split(np.array(winners), 2)

print('NORMAL\tNOISED')
for i in range(len(winners[0])):
    print(winners[0][i]._iid, '\t', winners[1][i]._iid)

print('Active neurons: {:d}'.format(numOfActiveNeurons))

from neurons import *
from emojis import *
from progressBar import *
import copy


""" Change emojis pixel values from (0,1) to (-1,1) """
def bipolar(emoji):
    for i in range(len(emoji)):
        if emoji[i] == 0:
            emoji[i] = -1
    return emoji

def countUniqueItems(arr):
    return len(Counter(arr).keys())

""" Get random but unique pixels and mult by -1 """
def noiseEmojis(arr, numOfNoisePixels):
    noisedArr = copy.deepcopy(arr)
    for emoji in noisedArr:
        pixels = np.random.choice(64, numOfNoisePixels, replace=False)
        for pixel in pixels:
            emoji[pixel] *= -1

    return noisedArr

""" Print emojis """
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



if __name__ == '__main__':

    #np.random.seed(5)
    neuronGrid = [15, 15]
    lRate=0.007         # learning rate
    fRate=0.418         # forget rate    BEST: 0.418
    numOfNoisePixels=10 # number of noised pixels
    epochs=200          # nuber of epochs
    decay=30*12         # decay for updating lRate during the learning proccess
    pBar = ProgressBar(length=56)

    emoji = Emoji()
    """ Get a set of training  emojis """
    emojisToGet = [ 'sad', 'smile', 'angry', 'laugh', 'surprised', 'confused' ]
    trainingSet = [ bipolar(emoji.getEmoji(name)) for name in emojisToGet ]

    """ Working well up to 9 noise pixels """
    noisedSet = noiseEmojis(trainingSet, numOfNoisePixels)

    drawEmojis(trainingSet)
    print('NOISED with {:d} pixels:\n'.format(numOfNoisePixels))
    drawEmojis(noisedSet)

    hebbGroup = HebbNeuronGroup(
        numOfInputs=64,
        numOfNeurons=neuronGrid,
        activFunc=Linear()(),
        lRateFunc=Linear()(),
        lRate=lRate,
        fRate=fRate
    )

    print('Running {:d} epochs...'.format(epochs))
    pBar.start(maxVal=epochs)
    for i in range(epochs):
        """ Will get winners from the latest epoch """
        winners1 = hebbGroup.train(trainingSet)
        pBar.update()

    """ Try to classify noised emojis """
    winners2 = hebbGroup.classify(noisedSet)

    numOfActiveNeurons = countUniqueItems(winners1+winners2)


    print('\t\tNORMAL\tNOISED')
    for i in range(len(winners1)):
        print('{}     \t{:d}\t{:d}'.format(emojisToGet[i], winners1[i]._iid, winners2[i]._iid))


    print('\nActive neurons: {:d}'.format(numOfActiveNeurons))
    print('lRate: {:.5f}'.format(hebbGroup._currentLRate))

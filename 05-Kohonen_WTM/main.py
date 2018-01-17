from math import ceil
from math import floor
from neurons import *
from data import *
from progressBar import *
import random
import copy
from prettytable import PrettyTable


def countUniqueItems(arr):
    return len(Counter(arr).keys())

def getMostCommonItem(arr):
    return Counter(arr).most_common(1)[0][0]


if __name__ == "__main__":

    """ Training parameters """
    epochs = 50
    decay = 0.1*(epochs)*13000
    neuronGrid = (20, 20)
    lRate = 0.11   # 0.07 one of the best
    neighbourhoodRadius = 3
    neighbourhoodRadiusMin = 0.5
    noNoisePixels = 2


    trainingData = Data()._letters
    testData = Data().getNoisedLetters(['U','N','C','O','P','Y','R','I','G','H','T','A','B','L','E'], noNoisePixels)
    numOfInputs = len(trainingData['A'])


    kohonenGroup = KohonenNeuronGroup(
        numOfInputs=numOfInputs,
        numOfNeurons=neuronGrid,
        processFunc=euklidesDistance,
        lRateFunc=Linear()(),
        neighbourhoodRadius=neighbourhoodRadius,
        nRadiusFunc=neighbourhoodRadiusCorrection(neighbourhoodRadius, neighbourhoodRadiusMin, epochs),
        lRate=lRate
    )

    paramsTable = PrettyTable()
    paramsTable.field_names = ['lRate', 'RadiusMax', 'RadiusMin', 'neurons', 'epochs']
    paramsTable.add_row([kohonenGroup._lRate, neighbourhoodRadius, neighbourhoodRadiusMin, kohonenGroup['totalNumOfNeurons'], epochs])
    print(paramsTable)


    print('\nrunning {:d} epochs...'.format(epochs))
    winners = {}
    pbar = ProgressBar()
    pbar.start(maxVal=epochs)
    for i in range(epochs):
        for key, value in trainingData.items():
            winners[key] = kohonenGroup.train(value)
        kohonenGroup.setNeighbourhoodRadius()
        pbar.update()

    testWinners = {}
    for key, value in testData.items():
        testWinners[key] = kohonenGroup.classify(value)


    trainingTable = PrettyTable()
    trainingTable.field_names = ['Letter', 'Neuron id', 'x', 'y', 'Neuron id*', 'x*', 'y*']
    for key, neuron in winners.items():
        testNeuron = testWinners.get(key, None)
        testNID = '' if not testNeuron else testNeuron._iid
        testX = '' if not testNeuron else testNeuron._x
        testY = '' if not testNeuron else testNeuron._y
        trainingTable.add_row([key, neuron._iid, neuron._x, neuron._y, testNID, testX, testY])
    uniqueNeurons = countUniqueItems(winners.values())
    print('\nActive neurons', uniqueNeurons)
    print('Number of letters', len(trainingData))
    uniqueNeurons = countUniqueItems(testWinners.values())
    print('\nActive test neurons', uniqueNeurons)
    print('Number of test letters', len(testData), '\n')
    print(trainingTable)
    print('* - testing letters noised with {:d} pixels.\n'.format(noNoisePixels))

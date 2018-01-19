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
    epochs = 100
    decay = 0.1*(epochs)*13000
    neuronGrid = (20, 20)
    lRate = 0.1   # 0.07 one of the best
    neighbourhoodRadius = 10
    neighbourhoodRadiusMin = 0.5
    noNoisePixels = 2
    assignmentMap = [[' ' for _ in range(neuronGrid[1])] for _ in range(neuronGrid[0])]
    resultsInNeighbourhood = 0
    resultsExact = 0
    resultsWrong = 0

    """ Getting training and test data """
    trainingData = Data()._letters
    testData = Data().getNoisedLetters(['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y', 'Z'], noNoisePixels)
    numOfInputs = len(trainingData['A'])

    """ Creating kohonen group with designated parameters """
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

    """ Process of learning """
    print('\nrunning {:d} epochs...'.format(epochs))
    winners = {}
    pbar = ProgressBar()
    pbar.start(maxVal=epochs)
    for i in range(epochs):
        for key, value in trainingData.items():
            winners[key] = kohonenGroup.train(value)
        kohonenGroup.setNeighbourhoodRadius()
        pbar.update()

    """ Process of testing """
    testWinners = {}
    for key, value in testData.items():
        testWinners[key] = kohonenGroup.classify(value)


    """ Printing results, comparision tables and map """
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

    for key, value in testWinners.items():
        dist = distance((value._x, value._y), (winners[key]._x, winners[key]._y))
        if dist <= neighbourhoodRadius and dist != 0:
            resultsInNeighbourhood+=1
        elif dist == 0:
            resultsExact += 1
    resultsWrong = len(testData)-(resultsExact+resultsInNeighbourhood)


    for key, value in winners.items():
        assignmentMap[value._x][value._y] = key

    for key, value in testWinners.items():
        assignmentMap[value._x][value._y] = key+'*'

    for row in assignmentMap:
        print(row)

    print('exact: {:d}\talmost: {:d}\twrong: {:d}'.format(resultsExact, resultsInNeighbourhood, resultsWrong))

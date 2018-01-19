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
    neighbourhoodRadius = 3
    neighbourhoodRadiusMin = 0.5
    noNoisePixels = 2
    assignmentMap = []


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


    for lRate in [0.9, 0.5, 0.1, 0.05, 0.01]:
        for radiusMax in [20, 10, 5, 4, 3, 2, 1]:
            kohonenGroup.resetGroup(lRate, radiusMax, neighbourhoodRadiusCorrection(radiusMax, neighbourhoodRadiusMin, epochs))
            winners = {}
            for i in range(epochs):
                for key, value in trainingData.items():
                    winners[key] = kohonenGroup.train(value)
                kohonenGroup.setNeighbourhoodRadius()
            testWinners = {}
            for key, value in testData.items():
                testWinners[key] = kohonenGroup.classify(value)
            resultsInNeighbourhood = 0
            resultsExact = 0
            uniqueNeurons = countUniqueItems(winners.values())
            for key, value in testWinners.items():
                dist = distance((value._x, value._y), (winners[key]._x, winners[key]._y))
                if dist <= radiusMax and dist != 0:
                    resultsInNeighbourhood+=1
                elif dist == 0:
                    resultsExact += 1
            resultsWrong = len(testData)-(resultsExact+resultsInNeighbourhood)
            print('lRate: {:.3f} radiusMax: {:d}\texact: {:d}\talmost: {:d}\twrong: {:d}\tunique: {:d}'.format(kohonenGroup._lRate, radiusMax, resultsExact, resultsInNeighbourhood, resultsWrong, uniqueNeurons))

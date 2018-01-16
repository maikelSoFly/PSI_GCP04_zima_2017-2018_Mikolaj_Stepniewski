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
    epochs = 10
    decay = 0.01*(epochs)*13000
    neuronGrid = [20, 20]
    lRate = 0.07    # 0.07 one of the best
    neighbourhoodRadius = 10
    neighbourhoodRadiusMin = 0.1


    trainingData = Data()._letters
    numOfInputs = len(trainingData['A'])


    kohonenGroup = KohonenNeuronGroup(
        numOfInputs=numOfInputs,
        numOfNeurons=neuronGrid,
        processFunc=euklidesDistance,
        lRateFunc=simpleLRateCorrection(decay),
        neighbourhoodRadius=neighbourhoodRadius,
        nRadiusFunc=neighbourhoodRadiusCorrection(neighbourhoodRadius, neighbourhoodRadiusMin, epochs),
        lRate=lRate
    )

    winners = {}
    for i in range(epochs):
        for key, value in trainingData.items():
            winners[key] = kohonenGroup.train(value)
        kohonenGroup.setNeighbourhoodRadius()

    print(winners)

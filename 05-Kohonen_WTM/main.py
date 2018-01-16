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
    decay = (epochs)*13000
    neuronGrid = [25, 25]
    lRate = 0.07    # 0.07 one of the best
    neighbourhoodRadius = 5
    neighbourhoodRadiusMin = 0.7


    trainingData = Data()._letters
    #testData = noiseLetters(trainingData)

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
    pbar = ProgressBar()
    pbar.start(maxVal=epochs)
    for i in range(epochs):
        for key, value in trainingData.items():
            winners[key] = kohonenGroup.train(value)
        kohonenGroup.setNeighbourhoodRadius()
        pbar.update()


    trainingTable = PrettyTable()
    trainingTable.field_names = ['Letter', 'Neuron iid', 'x', 'y']
    for key, neuron in winners.items():
        trainingTable.add_row([key, neuron._iid, neuron._x, neuron._y])
    uniqueNeurons = countUniqueItems(winners.values())
    print('\nActive neurons', uniqueNeurons)
    print('Number of letters', len(trainingData), '\n')
    print(trainingTable)

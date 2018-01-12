# @Author: Mikołaj Stępniewski <maikelSoFly>
# @Date:   2017-12-16T02:09:12+01:00
# @Email:  mikolaj.stepniewski1@gmail.com
# @Filename: main.py
# @Last modified by:   maikelSoFly
# @Last modified time: 2017-12-17T15:20:01+01:00
# @License: Apache License  Version 2.0, January 2004
# @Copyright: Copyright © 2017 Mikołaj Stępniewski. All rights reserved.



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

def averageParameters(species, n=50):
    sum = [0.0 for _ in range(4)]
    for row in species:
        sum[0] += row[0]
        sum[1] += row[1]
        sum[2] += row[2]
        sum[3] += row[3]
    return [ceil((sum[i]/n)*100)/100 for i in range(4)]


""" Main training function !!! """
def train(kohonenGroup, trainingData):
    pBar = ProgressBar()
    print('\n {} + {} + {}'.format(speciesNames[0], speciesNames[1], speciesNames[2]))
    pBar.start(maxVal=epochs)

    for i in range(epochs):
        testWinners = kohonenGroup.train(trainingData, histFreq=20)
        pBar.update()

    return testWinners





if __name__ == '__main__':

    """ Training parameters """
    epochs = 30
    decay = (epochs)*13000
    neuronGrid = [20, 20]
    lRate = 0.07    # 0.07 one of the best

    """ Exclude number of irises from total data set
    and add to test data """
    noExcludedIrises = 5

    dataUrl = 'http://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
    speciesNames = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']
    data = DataReader(url=dataUrl, delimiter=',').parse()
    testData = []

    for j in range(len(data)):
            data[j].pop()                           # remove species name
            data[j] = [float(i) for i in data[j]]   # cast str elements to float
            data[j] = normalizeInputs(data[j])      # normalize elements to 0...1 values

    irisDict = {'setosa': data[:50], 'versicolor': data[50:100], 'virginica': data[100:]}
    speciesArr = np.split(np.array(data), 3)


    """ Pop random irises from dict to testData """
    for i in range(noExcludedIrises):
        index = np.random.randint(50-i)
        testData.append(irisDict['setosa'].pop(index))
        testData.append(irisDict['versicolor'].pop(index))
        testData.append(irisDict['virginica'].pop(index))


    kohonenGroup = KohonenNeuronGroup(
        numOfInputs=4,
        numOfNeurons=neuronGrid,
        processFunc=euklidesDistance,
        lRateFunc=simpleLRateCorrection(decay),
        lRate=lRate
    )


    print('lRate0: {:.2f}\tdecay: {}\tneurons in group: {:d}\tepochs: {:d}'.format(
        kohonenGroup._lRate, decay, kohonenGroup['totalNumOfNeurons'], epochs
    ))

    print('\n•Averages:')
    for i, species in enumerate(speciesArr):
        print('{} \t{}'.format(averageParameters(species), speciesNames[i]))
    print()




    """ Training & testing """

    trainingData = []
    trainingData.extend(irisDict['setosa'])
    trainingData.extend(irisDict['versicolor'])
    trainingData.extend(irisDict['virginica'])

    trainingWinners = train(kohonenGroup, trainingData)
    numOfActiveNeurons = countUniqueItems(trainingWinners)
    trainingWinners = np.split(np.array(trainingWinners), 3)

    mostActiveNeurons1 = [getMostCommonItem(row) for row in trainingWinners]
    mostActiveNeurons = [getMostCommonItem(row)._iid for row in trainingWinners]
    print('\n\n•Training Summary:')
    table1 = PrettyTable()
    table1.field_names = ['Total active', 'Most active', 'Last lRate']
    table1.add_row([numOfActiveNeurons, mostActiveNeurons, kohonenGroup._currentLRate])
    print(table1)


    testWinners = kohonenGroup.classify(testData)

    testWinners = np.split(np.array(testWinners), len(testData)/3)

    print('\n\n•Test Results:')
    table = PrettyTable()
    table.field_names = [speciesNames[0], speciesNames[1], speciesNames[2]]
    for row in testWinners:
        table.add_row([neuron._iid for neuron in row ])

    print(table)


    for neuron in mostActiveNeurons1:
        print('\n')
        print('▄' * 25, '   [neuron: {:d}]\n\n'.format(neuron._iid))
        for row in neuron._weights:
            print(row)


    answ = input('Print error history?\ty/n: ')
    if answ == 'y':
        for neuron in mostActiveNeurons1:
            print('\n')
            print('▄' * 25, '   [neuron: {:d}]\n\n'.format(neuron._iid))
            for row in neuron._errorHist:
                print(row)

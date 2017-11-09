from letters import *
import numpy as np
from adaline import *

""" Mean Squared Error function """
def MSE(results,expected):
    sum = 0.0
    for i in range(len(results)):
        sum+=(results[i]-expected[i])**2
    return sum/len(results)

class InputVector:
    def __init__(self, x, d):
        self.__dict__['_x'] = x
        self.__dict__['_d'] = d
    def __getitem__(self, index):
        if index == 'x':
            return self._x
        if index == 'd':
            return self._d


if __name__ == "__main__":

    madaline = Madaline(
        3,                                      # number of neurons
        35,                                     # number of inputs
        Sign()(0.0),                            # activation function
        Madaline.ThresholdFuncType.MAJORITY     # threshold function
    )

    lmAdaline = LayerManager(
        2,                                      # number of layers
        [3, 1],                                 # number of neurons in layers
        [35, 3],                                # number of inputs in layers
        [Sign()(0.0), Sign()(0.0)],             # activation functions in layers
    )

    # 15 letters
    lettersInput = [
        LetterInput('a'),
        LetterInput('p'),
        LetterInput('o'),
        LetterInput('b'),
        LetterInput('A'),
        LetterInput('B'),
        LetterInput('C'),
        LetterInput('I'),
        LetterInput('F'),
        LetterInput('d'),
        LetterInput('c'),
        LetterInput('w'),
        LetterInput('H'),
        LetterInput('K'),
        LetterInput('D')
    ]

    print("Epoch", ",", "MSE error")
    aboveErr = True
    expectedForAllLetters = []
    for j in range(len(lettersInput)):
        expectedForAllLetters.extend(lettersInput[j]._interD)

    epoch = 0

    while(aboveErr):
        epochResults = []
        for j in range(len(lettersInput)):
            madaline._layer.trainNeurons(lettersInput[j]._x, lettersInput[j]._interD)

            results = lmAdaline.trainLayers([
                # for layer 0:
                InputVector(lettersInput[j]._x, lettersInput[j]._interD),
                # for layer 1:
                InputVector(lettersInput[j]._interD, lettersInput[j]._d)
            ])
            # result[0] is array of results from first layer
            epochResults.extend(results[0])

        mseVal = MSE(epochResults, expectedForAllLetters)
        if mseVal < 0.0001:
            aboveErr = False
        epoch += 1
        print(epoch, "," , mseVal)

    test = LetterInput('b')
    #print('Letter:', test._letter)
    #print("{}  ({})".format(madaline.process(test._x), madaline._thresholdFuncType.name))
    #print('3->1 Adalines', lmAdaline.processLayers(test._x))

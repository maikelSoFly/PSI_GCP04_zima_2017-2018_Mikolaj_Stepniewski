from neurons import *
from letters import *
import numpy as np

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

    ml = Multilayer(
        3,                                              # number of layers
        [3, 2, 1],                                      # number of neurons in layers
        [35, 3, 2],                                     # number of inputs in layers
        [Sigm()(1.0), Sigm()(1.0), Sign()(0.0)],        # activation functions in layers
        [
            Sigm().derivative(1.0),                     # activation function derivatives in layers
            Sigm().derivative(1.0),
            Sign().derivative()
        ]
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
        LetterInput('D')
    ]
    print("Epoch", ",", "MSE error")
    aboveErr = True
    expectedForAllLetters = []
    for j in range(len(lettersInput)):
        expectedForAllLetters.append(lettersInput[j]._d)

    epoch = 0
    results = []
    while(aboveErr):
        epochResults = []
        for j in range(len(lettersInput)):
            result = ml.trainLayers(
                InputVector(lettersInput[j]._x, lettersInput[j]._d)
            )
            # result[0] is array of results from first layer
            epochResults.extend(result)

        mseVal = MSE(epochResults, expectedForAllLetters)
        if mseVal < 0.0001:
            aboveErr = False
        epoch += 1
        print(epoch, "," , mseVal)


    test = LetterInput('K')
    print("Result:",",",ml.processLayers(test._x))

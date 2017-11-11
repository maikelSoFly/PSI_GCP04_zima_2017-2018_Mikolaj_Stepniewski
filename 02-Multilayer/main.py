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
        [35, 15, 1],                                    # number of neurons in layers
        [35, 35, 15],                                   # number of inputs in layers
        [Sign()(1.0), Sigm()(1.0), Sign()(0.0)],        # activation functions in layers
        [
            Sign().derivative(),                        # activation function derivatives in layers
            Sigm().derivative(1.0),
            Sign().derivative()
        ]
    )

    # TRAINING LETTERS
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
            # here result is the final answer from the net for certain letter
            epochResults.extend(result)

        mseVal = MSE(epochResults, expectedForAllLetters)
        if mseVal < 0.0001:
            aboveErr = False
        epoch += 1
        print(epoch, "," , mseVal)


    test = LetterInput('F')
    print("Letter:", test._letter)
    print("Result:", "UPPERCASE" if ml.processLayers(test._x) == 1 else "LOWERCASE")

from perceptron import *
from letters import *
import numpy as np

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
    #def __init__(self, nLayers, nNeurons, nInputs, activFuncs, activFuncDerivs):
    lm = LayerManager(2, [3, 1], [35, 3], [Sigm()(1.0), sign], [Sigm().derivative(1.0), one])

    lettersInput = [
        LetterInput('a'),
        LetterInput('p'),
        LetterInput('o'),
        LetterInput('b'),
        LetterInput('A'),
        LetterInput('B'),
        LetterInput('C'),
        LetterInput('D')
    ]

    for i in range(20):
        for j in range(len(lettersInput)):
            lm.trainLayers([
                InputVector(lettersInput[j]._x, lettersInput[j]._interD),
                InputVector(lettersInput[j]._interD, lettersInput[j]._d)
            ])

    test = LetterInput('t')
    print(lm.processLayers(test._x))

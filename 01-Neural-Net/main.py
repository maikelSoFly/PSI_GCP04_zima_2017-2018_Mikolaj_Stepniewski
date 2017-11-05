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

    lm = LayerManager(
        2,                                                 # number of layers
        [3, 1],                                            # number of neurons in layers
        [35, 3],                                           # number of inputs in layers
        [Sigm()(1.0), Sign()()],                           # activation functions in layers
        [Sigm().derivative(1.0), Sign().derivative()]      # activation function derivatives in layers
    )

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
                # for layer 0:
                InputVector(lettersInput[j]._x, lettersInput[j]._interD),
                # for layer 1:
                InputVector(lettersInput[j]._interD, lettersInput[j]._d)
            ])

    test = LetterInput('t')
    print(lm.processLayers(test._x))

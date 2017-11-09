from letters import *
import numpy as np
from adaline import *

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
        2,                                              # number of layers
        [3, 1],                                         # number of neurons in layers
        [35, 3],                                        # number of inputs in layers
        [Sign()(0.0), Sign()(0.0)],                     # activation functions in layers
    )

    lettersInput = [
        LetterInput('a'),
        LetterInput('p'),
        LetterInput('o'),
        LetterInput('b'),
        LetterInput('A'),
        LetterInput('B'),
        LetterInput('C'),
        LetterInput('I'),
        LetterInput('D')
    ]

    for i in range(20):
        for j in range(len(lettersInput)):
            madaline._layer.trainNeurons(lettersInput[j]._x, lettersInput[j]._interD)

            lmAdaline.trainLayers([
                # for layer 0:
                InputVector(lettersInput[j]._x, lettersInput[j]._interD),
                # for layer 1:
                InputVector(lettersInput[j]._interD, lettersInput[j]._d)
            ])

    test = LetterInput('b')
    print('Letter:', test._letter)
    print("{}  ({})".format(madaline.process(test._x), madaline._thresholdFuncType.name))
    print('3->1 Adalines', lmAdaline.processLayers(test._x))

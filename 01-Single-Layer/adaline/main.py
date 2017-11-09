from letters import *
import numpy as np
from adaline import *


if __name__ == "__main__":

    madaline = Madaline(
        3,                                      # number of neurons
        35,                                     # number of inputs
        Sign()(0.0),                            # activation function
        Madaline.ThresholdFuncType.MAJORITY     # threshold function
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
            madaline._layer.trainNeurons(lettersInput[j]._x, lettersInput[j]._interD)

    test = LetterInput('C')
    print('Letter:', test._letter)
    print("{}  ({})".format(madaline.process(test._x), madaline._thresholdFuncType.name))

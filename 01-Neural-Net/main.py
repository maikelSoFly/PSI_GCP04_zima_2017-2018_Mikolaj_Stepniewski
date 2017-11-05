from perceptron import *
from inputs import *
import numpy as np

if __name__ == "__main__":
    #def __init__(self, nLayers, nNeurons, nInputs, activFuncs, activFuncDerivs):
    ml = Multilayer(2, [3, 1], [35, 3], [Sigm()(1.0), sign], [Sigm().derivative(1.0), one])
    inputVectors = [
        TestInput('a'),
        TestInput('b'),
        TestInput('o'),
        TestInput('A'),
        TestInput('B'),
        TestInput('C'),
        TestInput('D')
    ]

    #print(inputVectors[0]._x)

    for i in range(15):
        for j in range(len(inputVectors)):
            ml.trainNet(inputVectors[j])

    ml.processNetOutput(inputVectors[0]._x)

from neuron import *
from letters import *
import numpy as np

""" Mean Squared Error function """
def MSE(results,expected):
    sum = 0.0
    for i in range(len(results)):
        sum+=(results[i]-expected[i])**2
    return sum/len(results)

""" Class made for encapsulating input data """
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

    """ Creating Sigmoidal layer with 3 neurons and Perceptron """
    lmSig = LayerManager(
        2,                                              # number of layers
        [3, 1],                                         # number of neurons in layers
        [35, 3],                                        # number of inputs in layers
        [Sigm()(1.0), Sign()(0.5)],                     # activation functions in layers
        [Sigm().derivative(1.0), Sign().derivative()],  # activation function derivatives in layers
    )

    # Training letters
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

    """ Creating array od expected values for certain pixels
        interD is 3 expected values for subtasks like:
            - does letter exeed left margin of grid?
            - does letter exeed right margin of grid?
            - does letter exeed top margin of grid?
    """
    for j in range(len(lettersInput)):
        expectedForAllLetters.extend(lettersInput[j]._interD)

    epoch = 0
    while(aboveErr):
        epochResults = []
        for j in range(len(lettersInput)):
            results = lmSig.trainLayers([
                # for layer 0:
                InputVector(lettersInput[j]._x, lettersInput[j]._interD),
                # for layer 1:
                InputVector(lettersInput[j]._interD, lettersInput[j]._d)
            ])
            # result[0] is array of results from first layer
            epochResults.extend(results[0])

        """ Calculating MSE error for every epoch"""
        mseVal = MSE(epochResults, expectedForAllLetters)
        if mseVal < 0.0001: # STOP IF MSE ERR IS LESS THAN 0.0001
            aboveErr = False
        epoch += 1
        print(epoch, "," , mseVal)

    """ Validation """
    test = LetterInput('w')
    print("Result:",",",lmSig.processLayers(test._x))

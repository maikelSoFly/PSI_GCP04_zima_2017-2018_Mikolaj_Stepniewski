from neurons import *
from rastrigin import *
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

    numOfDimensions = 2

    ml = Multilayer(
        2,                                              # number of layers
        [30, 1],                                    # number of neurons in layers
        [numOfDimensions, 30],                                   # number of inputs in layers
        [Sigm()(1.0), Linear()()],        # activation functions in layers
        [                       # activation function derivatives in layers

            Sigm().derivative(1.0),
            Linear().derivative(),
        ]
    )

    # TRAINING LETTERS
    numOfInputs = 100





    print("Epoch", ",", "MSE error")
    aboveErr = True
    epoch = 0
    results = []

    expectedForAllXs = []
    rastriginTrainer = RastriginInput(numOfDimensions, numOfInputs)
    for pt in rastriginTrainer._points:
        expectedForAllXs.append(pt._y)
    print(len(rastriginTrainer._points))

    while(aboveErr and epoch < 1000):
        epochResults = []
        for pt in rastriginTrainer._points:
            result = ml.trainLayers(
                InputVector(pt._xsArray, pt._y)
            )
            # here result is the final answer from the net for certain letter
            epochResults.append(result)

        mseVal = MSE(epochResults, expectedForAllXs)
        if mseVal < 10:
            aboveErr = False
        epoch += 1
        if epoch % 1 == 0:
            print(epoch, mseVal)


    print(ml.processLayers(rastriginTrainer._points[0]._xsArray), rastriginTrainer._points[0]._y)
    #
    # """ TESTING """
    # print('\n')
    # lettersInput.append(LetterInput('K')) # Letter unknown to the net
    # for letter in lettersInput:
    #     result = ml.processLayers(letter._x)
    #     strResult = 'lowercase' if result < 0.5 else 'UPPERCASE'
    #     print('Letter {}:\t{:.4}%\t{} '.format(
    #         letter._letter,
    #         100 * (1.0 - result) if result < 0.5 else 100 * result,
    #         strResult
    #     ))

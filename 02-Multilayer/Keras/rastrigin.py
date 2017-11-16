import numpy as np
import random

class RastriginInput:
    def __init__(self, numOfPoints=0, numOfDimensions=2):
        self.__dict__['_numOfPoints'] = numOfPoints
        self.__dict__['_numOfDimensions'] = numOfDimensions
        self.__dict__['_inputArray'] = None
        self.__dict__['_outputArray'] = None
        self.__dict__['_from'] = -2
        self.__dict__['_to'] = 2

    def initRastriginPointsRand(self, numOfPoints):
        """ Creating random x1, x2...xn and f(x1,x2..xn) as a training data
        for neural net """
        self._numOfPoints = numOfPoints
        inArr = []
        outArr = []
        for i in range(self._numOfPoints):
            xsArray = [random.uniform(-2,2), random.uniform(-2,2)]
            sum = 0
            for j in range(self._numOfDimensions):
                sum += (xsArray[j]**2) - 10 * np.cos(2 * np.pi * xsArray[j])
            y = 10 * self._numOfDimensions + sum
            inArr.append(xsArray)
            outArr.append(y)

        self._outputArray = outArr
        self._inputArray = inArr

    def initRastriginPoints(self, dx):
        """ Creating grid of x1, x2...xn and f(x1,x2..xn) with given dx as a training data
        for neural net """
        inArr = []
        outArr = []
        numOfPoints = int((np.absolute(self._from)+np.absolute(self._to)) / dx)
        x1 = self._from

        for i in range(numOfPoints+1):
            x2 = self._from
            for j in range(numOfPoints+1):
                xsArray = [x1, x2]
                inArr.append(xsArray)
                sum = 0
                for k in range(self._numOfDimensions):
                    sum += (xsArray[k]**2) - 10 * np.cos(2 * np.pi * xsArray[k])
                y = 10 * self._numOfDimensions + sum
                outArr.append(y)
                x2 += dx
            x1 += dx

        self._outputArray = outArr
        self._inputArray = inArr

    def getInputArray(self):
        return np.array(self._inputArray)

    def getOutputArray(self):
        return np.array(self._outputArray)


    """ Access method """
    def __getitem__(self, index):
        if index == 'from':
            return self._from
        elif index == 'to':
            return self._to
        elif index == 'numOfPoints':
            return self._numOfPoints
        elif index == 'numOfDimensions':
            return self._numOfDimensions



if __name__ == '__main__':

    ri = RastriginInput(3)
    ri.initRastriginPoints(0.5)
    print(ri.getInputArray(), ri.getOutputArray())

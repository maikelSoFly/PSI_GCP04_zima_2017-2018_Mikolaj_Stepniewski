import numpy as np

class RastriginPoint:
    def __init__(self, xsArray, y):
        self.__dict__['_xsArray'] = xsArray
        self.__dict__['_y'] = y
        self.__dict__['_noDimensions'] = len(xsArray)

    def __getitem__(self, index):
        if index == 'xsArray':
            return self._xsArray
        elif index == 'y':
            return self._y
        elif index == 'noDimensions':
            return self._noDimensions


class RastriginInput:
    def __init__(self, numOfDimensions, numOfPoints):
        self.__dict__['_numOfPoints'] = numOfPoints
        # if rand == False:
        #     self.__dict__['_points'] = self.initRastriginPoints(dx, numOfDimensions)
        # else:
        self.__dict__['_points'] = self.initRastriginPointsRand(numOfPoints, numOfDimensions)

    def initRastriginPointsRand(self, numOfPoints, numOfDimensions):
        dArray = []

        for i in range(numOfPoints):
            xsArray = np.random.uniform(-5.12, 5.12, numOfDimensions)
            sum = 0
            for j in range(numOfDimensions):
                sum += (xsArray[j]**2) - 10 * np.cos(2 * np.pi * xsArray[j])
            y = 10 * numOfDimensions + sum

            dArray.append(RastriginPoint(xsArray, y))

        return dArray

    def initRastriginPoints(self, dx, numOfDimensions):
        dArray = []
        numOfPoints = int(round(10.24 / dx))
        x1 = -5.12
        for i in range(numOfPoints+1):
            for j in range(numOfPoints+1):
                x2 = -5.12
                xsArray = [x1, x2]
                sum = 0
                for k in range(numOfDimensions):
                    sum += (xsArray[k]**2) - 10 * np.cos(2 * np.pi * xsArray[k])
                y = 10 * numOfDimensions + sum
                dArray.append(RastriginPoint(xsArray, y))
                x2 += dx
            x1 += dx

        return dArray

    """ Access method """
    def __getitem__(self, index):
        if index == 'points':
            return self._points



if __name__ == '__main__':

    ri = RastriginInput(2, 300)
    for point in ri._points:
        print('xs: ', point._xsArray, '\ty: ', point._y)

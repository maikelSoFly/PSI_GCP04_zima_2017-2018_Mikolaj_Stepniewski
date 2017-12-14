import numpy as np
from math import sqrt

""" Sign function which can be translated by given value. Used as
    activation function for perceptron.

    - Parameters:
        - translation: breaking point for the function.

    - Usage:
        - Sign()(0.5)
            - returns sign function for unipolar sigmoidal function
"""
class Sign:
    def __call__(self, translation):
        def sign(x):
            if x < translation:
                return 0
            else:
                return 1
        return sign

    def derivative(self):
        def signDeriv(x):
            return 1
        return signDeriv


""" Sigmoidal function & its derivative for given beta. Used as
    Activation function for perceptron.

    - Parameters:
        - beta: sigmoidal function parameter. Its value affects
        function shape. The greater the value the steeper is the function.

    - Usage:
        - Sigm()(0.5)
            - returns: sigm(x) function with beta=0.5
        - Sigm().derivative(0.5)
            - returns sigmDeriv function
            with beta=0.5
"""
class Sigm:
    def __call__(self, beta):
        def sigm(x):
            return 1.0/(1.0+np.exp(-beta*x))
        sigm.__name__ += '({0:.3f})'.format(beta)
        return sigm
    def derivative(self, beta):
        def sigmDeriv(x):
            return beta*np.exp(-beta*x)/((1.0+np.exp(-beta*x))**2)
        sigmDeriv.__name__ += '({0:.3f})'.format(beta)
        return sigmDeriv

class SignSigm:
    def __call__(self,alfa):
        def signSigm(x):
            return (2.0/(1.0+np.exp(-alfa*x)))-1.0
        signSigm.__name__+='({0:.3f})'.format(alfa)
        return signSigm
    def derivative(self,alfa):
        def signSigmDeriv(x):
            return 2.0*alfa*np.exp(-alfa*x)/((1.0+np.exp(-alfa*x))**2)
        signSigmDeriv.__name__+='({0:.3f})'.format(alfa)
        return signSigmDeriv

def hardSign(x):
    if x<0:
        return -1.0
    return 1.0

class Linear:
    def __call__(self):
        def linear(x):
            return x
        return linear
    def derivative(self):
        def linearDeriv(x):
            return 1
        return linearDeriv

""" Mean Squared Error function """
def MSE(results, expected):
    sum = 0.0
    for i in range(len(results)):
        sum+=(results[i]-expected[i])**2
    return sum/len(results)

def euklidesDistance(v1, v2):
    sum = 0.0
    if len(v1) != len(v2):
        raise Exception('\t[!]\tLenghts of vectors are not equal.')
    else:
        for i in range(len(v1)):
            sum += (v1[i] - v2[i])**2
    return sqrt(sum)

def simpleLearnCorrection(lambd):
    t=-1
    def f():
        nonlocal t
        t+=1
        return np.exp(-t/lambd)
    return f

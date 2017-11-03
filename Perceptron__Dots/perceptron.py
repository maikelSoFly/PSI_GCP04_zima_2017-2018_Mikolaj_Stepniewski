import random
from consts import *

def sign(x):
    if x >= 0:
        return 1
    else:
        return -1

class Perceptron:
    """ Perceptron is a simple neural net that can
        specify which class object belongs to. """

    weights = []
    training_counter = 0

    def __init__(self, n):
        for i in range(n):
            self.weights.append(random.uniform(-1,1))

    def guess(self, inputs):
        sum = 0.0
        for i in range(len(self.weights)):
            sum += inputs[i] * self.weights[i]

        """ Return output """
        return sign(sum)

    def train(self, inputs, target):
        guess = self.guess(inputs)
        error = target - guess

        for i in range(len(self.weights)):
            self.weights[i] += error * inputs[i] * LR

    def guess_y(self, x):
        w0 = self.weights[0] # x
        w1 = self.weights[1] # y
        w2 = self.weights[2] # bias

        return -(w0/w1) * x - (w2/w1)

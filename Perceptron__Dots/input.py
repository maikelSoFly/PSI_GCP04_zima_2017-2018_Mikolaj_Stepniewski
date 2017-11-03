import random
from consts import *
from graphics import *
from numpy import interp

""" In carthesian coordinates """
def f(x):
    return -0.1 * x - 0.2


class InputPoint:
    x = 0.0
    y = 0.0
    bias = 1
    label = None

    def interpX(self):
        return interp(self.x, [-1,1], [0,WIDTH])

    def interpY(self):
        return interp(self.y, [-1,1], [HEIGHT,0])

    def __init__(self, point):
        if point == None:
            self.x = random.uniform(-1, 1)
            self.y = random.uniform(-1, 1)
        else:
            self.x = point.x
            self.y = point.y

        interpX = self.interpX()
        interpY = self.interpY()

        if self.y > f(self.x):
            self.label = 1
        else:
            self.label = -1

    def show(self, win):
        circle = Circle(Point(self.interpX(), self.interpY()), 5)
        if self.label == 1:
            circle.setFill('blue')
        else :
            circle.setFill('yellow')
        circle.draw(win)

import random

""" -1 is False, 1 is True """

class InputPoint:
    x = -1
    y = -1
    label = -1

    def convertToBool(self, num):
        return True if num == 1 else False

    def __init__(self, randomize=False):
        if randomize == True:
            rndX = random.uniform(-1,1)
            rndY = random.uniform(-1,1)
            if rndX > 0:
                self.x = 1
            if rndY > 0:
                self.y = 1

            """ Main condition. """
            if self.convertToBool(self.x) and self.convertToBool(self.y):
                self.label = 1
            else:
                self.label = -1

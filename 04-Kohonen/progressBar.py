from math import ceil
from math import floor

class ProgressBar:
    def __init__(self, length):
        self.__dict__['_l'] = length
        self.__dict__['_x'] = None
        self.__dict__['_t'] = None
        self.__dict__['_mod'] = None
        self.__dict__['_times'] = None


    def start(self, maxVal):
        self._x = maxVal
        self._t = -1

        # 69 is TODO :)
        self._mod = ceil(self._x/self._l) if self._x > self._l else ceil(self._x/self._l)
        self._times = 1 if self._x > self._l else floor(self._l/self._x)
        self._restTimes = self._l-floor((self._x-1)/self._mod) if self._x > self._l else self._l-(self._x-1)*floor(self._l/self._x)

        print('▁ ' * self._l, end = '', flush=True)
        print()

    def update(self):
        self._t += 1
        if self._t >= self._x:
            raise Exception('ProgressBar:\titerator is out of bounds.')

        if self._t != 0 and self._t % self._mod == 0:
            print('▇ ' * self._times, end='', flush=True)
        if self._t == self._x-1:
            print('▇ ' * self._restTimes, end = '')

# @Author: Mikołaj Stępniewski <maikelSoFly>
# @Date:   2017-12-16T10:40:44+01:00
# @Email:  mikolaj.stepniewski1@gmail.com
# @Filename: progressBar.py
# @Last modified by:   maikelSoFly
# @Last modified time: 2017-12-16T13:20:50+01:00
# @Copyright: Copyright © 2017 Mikołaj Stępniewski. All rights reserved.



from math import ceil
from math import floor
import time

class ProgressBar:
    def __init__(self, length):
        self.__dict__['_l'] = length
        self.__dict__['_x'] = None
        self.__dict__['_t'] = None
        self.__dict__['_mod'] = None
        self.__dict__['_times'] = None
        self.__dict__['_lineChar'] = '_'
        self.__dict__['_char'] = '▋'
        self.__dict__['_startTime'] = None
        self.__dict__['_elapsedTime'] = None


    def start(self, maxVal):
        self._x = maxVal
        self._t = -1

        self._mod = ceil(self._x/self._l) if self._x > self._l else ceil(self._x/self._l)
        self._times = 1 if self._x > self._l else floor(self._l/self._x)
        self._restTimes = self._l-floor((self._x-1)/self._mod) if self._x > self._l else self._l-(self._x-1)*floor(self._l/self._x)

        print(self._lineChar * self._l, end = '', flush=True)
        print()

        self._startTime = time.time()
        self._elapsedTime = None

    def update(self):
        self._t += 1
        if self._t >= self._x:
            raise Exception('ProgressBar:\titerator is out of bounds.')

        if self._t != 0 and self._t % self._mod == 0:
            print(self._char * self._times, end='', flush=True)
        if self._t == self._x-1:
            self._elapsedTime = time.time() - self._startTime
            print(self._char * self._restTimes, end = '')
            print('   DONE\t({:.3f} s)'.format(self._elapsedTime))

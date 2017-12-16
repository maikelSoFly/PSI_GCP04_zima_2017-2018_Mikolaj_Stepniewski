# @Author: Mikołaj Stępniewski <maikelSoFly>
# @Date:   2017-12-12T17:14:43+01:00
# @Email:  mikolaj.stepniewski1@gmail.com
# @Filename: neuron.py
# @Last modified by:   maikelSoFly
# @Last modified time: 2017-12-16T13:21:08+01:00
# @Copyright: Copyright © 2017 Mikołaj Stępniewski. All rights reserved.



import numpy as np

class Neuron:
    def __init__(self, numOfInputs, iid, activFunc, lRate=0.01, bias=-0.5):
        self._weights = np.array([np.random.uniform(0, 1) for _ in range(numOfInputs)])
        self.__dict__['_activFunc'] = activFunc
        self.__dict__['_bias'] = bias
        self.__dict__['_lRate'] = lRate
        self.__dict__['_error'] = None
        self.__dict__['_sum'] = None
        self.__dict__['_val'] = None
        self.__dict__['_iid'] = iid

    def process(self, inputs):
        self._sum = np.dot(self._weights, inputs) + self._bias
        return self._activFunc(self._sum)

    def setTrainingData(self, data):
        for inputs in data:
            if len(inputs) != len(self._weights):
                raise Exception('Different number of weights and training set!')
        else:
            self._trainingData = data

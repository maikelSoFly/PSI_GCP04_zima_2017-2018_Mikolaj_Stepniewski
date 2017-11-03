class InputVector:
    def __init__(self, x):
        self.__dict__['_x'] = x
        if x[0] == 1 and x[1] == 1:
            self.__dict__['_d'] = 1
        else:
            self.__dict__['_d'] = 0

    def __getitem__(self,index):
        if index=='d':
            return self._d
        if index=='x':
            return self._x

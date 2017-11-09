class LetterInput():
    def __init__(self, letter):
        self.__dict__['_x'] = []
        self.__dict__['_d'] = None
        self.__dict__['_interD'] = None
        self.__dict__['_letter'] = letter
        self.getLetter()

    def getLetter(self):
        if self._letter == 'a':
            self._x = [
                 -1,  -1,  -1,  -1,  -1,
                 -1,  -1,  -1,  -1,  -1,
                 -1,  -1,  -1,  -1,  -1,
                 -1,  -1,  1,  1,  -1,
                 -1,  1,  -1,  1,  -1,
                 -1,  1,  -1,  1,  -1,
                 -1,  1,  1,  1,  1
                ]
            self._interD = [-1,-1,-1]
            self._d = -1

        elif self._letter == 'I':
            self._x = [
                -1,  -1,  1,  -1,  -1,
                -1,  -1,  1,  -1,  -1,
                -1,  -1,  1,  -1,  -1,
                -1,  -1,  1,  -1,  -1,
                -1,  -1,  1,  -1,  -1,
                -1,  -1,  1,  -1,  -1,
                -1,  -1,  1,  -1,  -1
            ]
            self._interD = [-1,1,-1]
            self._d = 1


        elif self._letter == 'b':
            self._x = [
                1,  -1,  -1,  -1,  -1,
                1,  -1,  -1,  -1,  -1,
                1,  -1,  -1,  -1,  -1,
                1,  1,  1,  1,  -1,
                1,  -1,  -1,  1,  -1,
                1,  -1,  -1,  1,  -1,
                1,  1,  1,  1,  -1
                ]
            self._interD = [1,-1,-1]
            self._d = -1

        elif self._letter == 't':
            self._x = [
                -1,  -1,  -1,  -1,  -1,
                -1,  -1,  -1,  -1,  -1,
                -1,  -1,  -1,  -1,  -1,
                -1,  -1,  1,  -1,  -1,
                -1,  1,  1,  1,  -1,
                -1,  -1,  1,  -1,  -1,
                -1,  -1,  1,  1,  -1
                ]
            self._interD = [-1,-1,-1]
            self._d = -1

        elif self._letter == 'p':
            self._x = [
                -1,  -1,  -1,  -1,  -1,
                -1,  -1,  -1,  -1,  -1,
                -1,  -1,  -1,  -1,  -1,
                -1,  1,  1,  1,  -1,
                -1,  1,  -1,  1,  -1,
                -1,  1,  1,  1,  -1,
                -1,  1,  -1,  1,  -1
                ]
            self._interD = [-1,-1,-1]
            self._d = -1

        elif self._letter == 'o':
            self._x = [
                 -1,  -1,  -1,  -1,  -1,
                 -1,  -1,  -1,  -1,  -1,
                 -1,  -1,  -1,  -1,  -1,
                 -1,  1,  1,  1,  -1,
                 -1,  1,  -1,  1,  -1,
                 -1,  1,  -1,  1,  -1,
                 -1,  1,  1,  1,  -1
                ]
            self._interD = [-1,-1,-1]
            self._d = -1

        elif self._letter == 'A':
            self._x = [
                 -1,  1,  1,  1,  -1,
                 1,  -1,  -1,  -1,  1,
                 1,  -1,  -1,  -1,  1,
                 1,  1,  1,  1,  1,
                 1,  -1,  -1,  -1,  1,
                 1,  -1,  -1,  -1,  1,
                 1,  -1,  -1,  -1,  1
                ]
            self._interD = [1,1,1]
            self._d = 1

        elif self._letter == 'B':
            self._x = [
                1,  1,  1,  1,  -1,
                1,  -1,  -1,  -1,  1,
                1,  -1,  -1,  -1,  1,
                1,  1,  1,  1,  -1,
                1,  -1,  -1,  -1,  1,
                1,  -1,  -1,  -1,  1,
                1,  1,  1,  1,  -1
            ]
            self._interD = [1,1,1]
            self._d = 1


        elif self._letter == 'C':
            self._x = [
                -1,  1,  1,  1,  -1,
                1,  -1,  -1,  -1,  1,
                1,  -1,  -1,  -1,  -1,
                1,  -1,  -1,  -1,  -1,
                1,  -1,  -1,  -1,  -1,
                1,  -1,  -1,  -1,  1,
                -1,  1,  1,  1,  -1,
            ]
            self._interD = [1,1,-1]
            self._d = 1

        # 1 + 1 - 1 = 1
        # OR: 3 - 1 = 2
        # if x > -2 ret 1

        elif self._letter == 'D':
            self._x= [
                1,  1,  1,  1,  -1,
                1,  -1,  -1,  -1,  1,
                1,  -1,  -1,  -1,  1,
                1,  -1,  -1,  -1,  1,
                1,  -1,  -1,  -1,  1,
                1,  -1,  -1,  -1,  1,
                1,  1,  1,  1,  -1,
            ]
            self._interD = [1,1,1]
            self._d = 1
    def __getitem__(self, index):
        if index=='x':
            return self._x
        elif index=='d':
            return self._d
        elif index=='interD':
            return self._interD
        elif index=='letter':
            return self._letter

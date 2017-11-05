class TestInput():
    """docstring forTestInput."""
    """
        Test input is a vector which represents capital and small letters like as in
        5x7 table
    """
    #x = []
    availableLetters = ['a', 'b', 'o', 'A', 'B', 'C', 'D']
    def __init__(self, letter):
        self.__dict__['_x'] = []
        self.__dict__['_d'] = None
        self.__dict__['_letterOfTest'] = letter

        self.getLetter()
        #print(self._x)


    def getLetter(self):
        if self._letterOfTest == 'a':
            self._x = [
                 0,  0,  0,  0,  0,
                 0,  0,  0,  0,  0,
                 0,  0,  0,  0,  0,
                 0,  0,  1,  1,  0,
                 0,  1,  0,  1,  0,
                 0,  1,  0,  1,  0,
                 0,  1,  1,  1,  1
                ]
            self._d = 0


        elif self._letterOfTest == 'b':
            self._x = [
                1,  0,  0,  0,  0,
                1,  0,  0,  0,  0,
                1,  0,  0,  0,  0,
                1,  1,  1,  1,  0,
                1,  0,  0,  1,  0,
                1,  0,  0,  1,  0,
                1,  1,  1,  1,  0
                ]
            self._d = 0

        elif self._letterOfTest == 'o':
            self._x = [
                 0,  0,  0,  0,  0,
                 0,  0,  0,  0,  0,
                 0,  0,  0,  0,  0,
                 0,  1,  1,  1,  0,
                 0,  1,  0,  1,  0,
                 0,  1,  0,  1,  0,
                 0,  1,  1,  1,  0
                ]
            self._d = 0

        elif self._letterOfTest == 'A':
            self._x = [
                 0, 1,  1,  1,  0,
                 1,  0,  0,  0,  1,
                 1,  0,  0,  0,  1,
                 1,  1,  1,  1,  1,
                 1,  0,  0,  0, 1,
                 1,  0,  0,  0, 1,
                 1,  0,  0,  0,  1
                ]
            self._d = 1

        elif self._letterOfTest == 'B':
            self._x = [
                1,  1,  1,  1,  0,
                1,  0,  0,  0,  1,
                1,  0,  0,  0,  1,
                1,  1,  1,  1,  0,
                1,  0,  0,  0,  1,
                1,  0,  0,  0,  1,
                1,  1,  1,  1,  0
            ]
            self._d = 1


        elif self._letterOfTest == 'C':
            self._x = [
                0, 1,  1,  1,  0,
                1,  0,  0,  0,  1,
                1,  0,  0,  0,  0,
                1,  0,  0,  0,  0,
                1,  0,  0,  0,  0,
                1,  0,  0,  0,  1,
                0, 1,  1,  1,  0,
            ]
            self._d = 1


        elif self._letterOfTest == 'D':
            self._x= [
                1,  1,  1,  1,  0,
                1,  0,  0,  0,  1,
                1,  0,  0,  0,  1,
                1,  0,  0,  0,  1,
                1,  0,  0,  0,  1,
                1,  0,  0,  0,  1,
                1,  1,  1,  1,  0,
            ]
            self._d = 1
    def __getitem__(self, index):
        if index=='x':
            return self._x
        elif index=='d':
            return self._d

    # def makeTestInputs(no_of_tests):
    #     testInputsArray = []
    #     x = 0
    #     for i in range(0, no_of_tests):
    #         testInputsArray.append(TestInput(TestInput.availableLetters[x]))
    #         x += 1
    #         if x == len(TestInput.availableLetters):
    #             x = 0
    #     return testInputsArray

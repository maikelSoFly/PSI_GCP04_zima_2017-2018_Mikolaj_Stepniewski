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
                 0,  0,  0,  0,  0,
                 0,  0,  0,  0,  0,
                 0,  0,  0,  0,  0,
                 0,  0,  1,  1,  0,
                 0,  1,  0,  1,  0,
                 0,  1,  0,  1,  0,
                 0,  1,  1,  1,  1
                ]
            self._interD = [0,0,0]
            self._d = 0


        elif self._letter == 'b':
            self._x = [
                1,  0,  0,  0,  0,
                1,  0,  0,  0,  0,
                1,  0,  0,  0,  0,
                1,  1,  1,  1,  0,
                1,  0,  0,  1,  0,
                1,  0,  0,  1,  0,
                1,  1,  1,  1,  0
                ]
            self._interD = [1,0,0]
            self._d = 0

        elif self._letter == 't':
            self._x = [
                0,  0,  0,  0,  0,
                0,  0,  0,  0,  0,
                0,  0,  0,  0,  0,
                0,  0,  1,  0,  0,
                0,  1,  1,  1,  0,
                0,  0,  1,  0,  0,
                0,  0,  1,  1,  0
                ]
            self._interD = [0,0,0]
            self._d = 0

        elif self._letter == 'p':
            self._x = [
                0,  0,  0,  0,  0,
                0,  0,  0,  0,  0,
                0,  0,  0,  0,  0,
                0,  1,  1,  1,  0,
                0,  1,  0,  1,  0,
                0,  1,  1,  1,  0,
                0,  1,  0,  1,  0
                ]
            self._interD = [0,0,0]
            self._d = 0

        elif self._letter == 'o':
            self._x = [
                 0,  0,  0,  0,  0,
                 0,  0,  0,  0,  0,
                 0,  0,  0,  0,  0,
                 0,  1,  1,  1,  0,
                 0,  1,  0,  1,  0,
                 0,  1,  0,  1,  0,
                 0,  1,  1,  1,  0
                ]
            self._interD = [0,0,0]
            self._d = 0

        elif self._letter == 'A':
            self._x = [
                 0,  1,  1,  1,  0,
                 1,  0,  0,  0,  1,
                 1,  0,  0,  0,  1,
                 1,  1,  1,  1,  1,
                 1,  0,  0,  0,  1,
                 1,  0,  0,  0,  1,
                 1,  0,  0,  0,  1
                ]
            self._interD = [1,1,1]
            self._d = 1

        elif self._letter == 'B':
            self._x = [
                1,  1,  1,  1,  0,
                1,  0,  0,  0,  1,
                1,  0,  0,  0,  1,
                1,  1,  1,  1,  0,
                1,  0,  0,  0,  1,
                1,  0,  0,  0,  1,
                1,  1,  1,  1,  0
            ]
            self._interD = [1,1,1]
            self._d = 1


        elif self._letter == 'C':
            self._x = [
                0,  1,  1,  1,  0,
                1,  0,  0,  0,  1,
                1,  0,  0,  0,  0,
                1,  0,  0,  0,  0,
                1,  0,  0,  0,  0,
                1,  0,  0,  0,  1,
                0,  1,  1,  1,  0,
            ]
            self._interD = [1,1,0]
            self._d = 1


        elif self._letter == 'D':
            self._x= [
                1,  1,  1,  1,  0,
                1,  0,  0,  0,  1,
                1,  0,  0,  0,  1,
                1,  0,  0,  0,  1,
                1,  0,  0,  0,  1,
                1,  0,  0,  0,  1,
                1,  1,  1,  1,  0,
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

    # def makeTestInputs(no_of_tests):
    #     testInputsArray = []
    #     x = 0
    #     for i in range(0, no_of_tests):
    #         testInputsArray.append(TestInput(TestInput.availableLetters[x]))
    #         x += 1
    #         if x == len(TestInput.availableLetters):
    #             x = 0
    #     return testInputsArray

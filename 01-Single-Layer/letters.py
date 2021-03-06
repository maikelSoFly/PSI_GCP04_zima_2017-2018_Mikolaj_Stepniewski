class LetterInput():
    def __init__(self, letter):
        self.__dict__['_x'] = []
        # expected output for whole task (lowercase/uppercase)
        self.__dict__['_d'] = None
        self.__dict__['_interD'] = None  # expected outputs for 3 subtasks
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
            self._interD = [0, 0, 0]
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
            self._interD = [1, 0, 0]
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
            self._interD = [0, 0, 0]
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
            self._interD = [0, 0, 0]
            self._d = 0

        elif self._letter == 'c':
            self._x = [
                0,  0,  0,  0,  0,
                0,  0,  0,  0,  0,
                0,  0,  0,  0,  0,
                0,  1,  1,  1,  0,
                0,  1,  0,  0,  0,
                0,  1,  0,  0,  0,
                0,  1,  1,  1,  0
            ]
            self._interD = [0, 0, 0]
            self._d = 0

        elif self._letter == 'w':
            self._x = [
                0,  0,  0,  0,  0,
                0,  0,  0,  0,  0,
                0,  0,  0,  0,  0,
                1,  0,  0,  0,  1,
                1,  0,  0,  0,  1,
                1,  0,  1,  0,  1,
                0,  1,  0,  1,  0
            ]
            self._interD = [1, 0, 1]
            self._d = 0

        elif self._letter == 'd':
            self._x = [
                0,  0,  0,  0,  1,
                0,  0,  0,  0,  1,
                0,  0,  0,  0,  1,
                0,  0,  1,  1,  1,
                0,  1,  0,  0,  1,
                0,  1,  0,  0,  1,
                0,  1,  1,  1,  1
            ]
            self._interD = [0, 0, 0]
            self._d = 0

        elif self._letter == 'o':
            self._x = [
                0, 0, 0, 0, 0,
                0, 0, 0, 0, 0,
                0, 0, 0, 0, 0,
                0, 1, 1, 1, 0,
                0, 1, 0, 1, 0,
                0, 1, 0, 1, 0,
                0, 1, 1, 1, 0
            ]
            self._interD = [0, 0, 0]
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
            self._interD = [1, 1, 1]
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
            self._interD = [1, 1, 1]
            self._d = 1

        elif self._letter == 'I':
            self._x = [
                0,  0,  1,  0,  0,
                0,  0,  1,  0,  0,
                0,  0,  1,  0,  0,
                0,  0,  1,  0,  0,
                0,  0,  1,  0,  0,
                0,  0,  1,  0,  0,
                0,  0,  1,  0,  0
            ]
            self._interD = [0, 1, 0]
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
            self._interD = [1, 1, 0]
            self._d = 1

        elif self._letter == 'D':
            self._x = [
                1,  1,  1,  1,  0,
                1,  0,  0,  0,  1,
                1,  0,  0,  0,  1,
                1,  0,  0,  0,  1,
                1,  0,  0,  0,  1,
                1,  0,  0,  0,  1,
                1,  1,  1,  1,  0,
            ]
            self._interD = [1, 1, 1]
            self._d = 1

        elif self._letter == 'F':
            self._x = [
                1, 1, 1, 1, 1,
                1, 0, 0, 0, 0,
                1, 0, 0, 0, 0,
                1, 1, 1, 1, 0,
                1, 0, 0, 0, 0,
                1, 0, 0, 0, 0,
                1, 0, 0, 0, 0,
            ]
            self._interD = [1,  1,  0]
            self._d = 1

        elif self._letter == 'K':
            self._x = [
                1, 0, 0, 0, 1,
                1, 0, 0, 1, 0,
                1, 0, 1, 0, 0,
                1, 1, 0, 0, 0,
                1, 0, 1, 0, 0,
                1, 0, 0, 1, 0,
                1, 0, 0, 0, 1,
            ]
            self._interD = [1,  1,  1]
            self._d = 1

        elif self._letter == 'H':
            self._x = [
                1, 0, 0, 0, 1,
                1, 0, 0, 0, 1,
                1, 0, 0, 0, 1,
                1, 1, 1, 1, 1,
                1, 0, 0, 0, 1,
                1, 0, 0, 0, 1,
                1, 0, 0, 0, 1,
            ]
            self._interD = [1,  1,  1]
            self._d = 1

    def __getitem__(self, index):
        if index == 'x':
            return self._x
        elif index == 'd':
            return self._d
        elif index == 'interD':
            return self._interD

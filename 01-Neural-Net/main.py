from perceptron import *
from inputs import *
import numpy as np

if __name__ == "__main__":
    inputs = [
        InputVector([0,0,1]),
        InputVector([0,1,1]),
        InputVector([1,0,1]),
        InputVector([1,1,1])
    ]

    w=[np.random.ranf()*np.random.choice([0,1]) for _ in range(3)]
    activFunc = Sigm()(1.0)
    activFuncDeriv = Sigm().derivative(1.0)


    p = Perceptron(w, activFunc, activFuncDeriv)

    for i in range(500):
        p.train(inputs[0]._x, 0)
        p.train(inputs[1]._x, 1)
        p.train(inputs[2]._x, 0)
        p.train(inputs[3]._x, 1)



    print("Guess:", p.process(inputs[0]._x))
    print("Guess:", p.process(inputs[1]._x))
    print("Guess:", p.process(inputs[2]._x))
    print("Guess:", p.process(inputs[3]._x))

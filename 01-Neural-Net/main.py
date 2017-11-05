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

    w1=[np.random.ranf() for _ in range(3)]
    w2=[np.random.ranf() for _ in range(3)]
    w3=[np.random.ranf() for _ in range(2)]

    activFunc = Sigm()(1.0)
    activFuncDeriv = Sigm().derivative(1.0)


    p1 = Perceptron(w1, activFunc, activFuncDeriv)
    p2 = Perceptron(w2, activFunc, activFuncDeriv)
    p3 = Perceptron(w3, ident, activFuncDeriv)

    for i in range(500):
        p1.train(inputs[0]._x, 0)
        p1.train(inputs[1]._x, 1)
        p1.train(inputs[2]._x, 0)
        p1.train(inputs[3]._x, 1)
        p2.train(inputs[0]._x, 0)
        p2.train(inputs[1]._x, 1)
        p2.train(inputs[2]._x, 0)
        p2.train(inputs[3]._x, 1)
        p3.train([1,1], 1)
        p3.train([0,0], 0)

    print("Guess:", p3.process( [p1.process(inputs[1]._x), p2.process(inputs[1]._x) ] ))

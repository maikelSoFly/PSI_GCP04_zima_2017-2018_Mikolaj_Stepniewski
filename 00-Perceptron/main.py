from input import *
from perceptron import *

training_inputs = []
p = Perceptron(3)

def main():
    eras_count = 20
    training_inputs_count = 10
    fail_counter = 0
    test_range = 50
    test_inputs = []

    """ Make random inputs [x:Bool, y:Bool]. """
    for i in range(training_inputs_count):
        training_inputs.append(InputPoint(randomize=True))


    for i in range(test_range):
        test_inputs.append(InputPoint(randomize=True))


    """ Train perceptron based on that inputs. """
    print()
    for i in range(eras_count):
        avg_err = 0
        fail_counter = 0
        for input in training_inputs[:]:
            inputs = [input.x, input.y, BIAS]
            target = input.label
            error = p.train(inputs, target)
            avg_err += error * error

        avg_err *= 0.5
        for j in range(test_range):
            testInput = test_inputs[j]
            inputs = [testInput.x, testInput.y, BIAS]
            guess = p.guess(inputs)
            if testInput.label != guess:
                fail_counter += 1
        print('era: %s, avg_err: %d, test errors: %d/%d' % (i, avg_err, fail_counter, len(test_inputs)))



    print('\n\n')
    

    """ Make some new randomized inputs & check if it really works. """
    # test_p1 = InputPoint()
    # for i in range(test_range):
    #     testInput = InputPoint(randomize=True)
    #     inputs = [testInput.x, testInput.y, BIAS]
    #     guess = p.guess(inputs)
    #     strr = ('True ' if testInput.x == 1 else 'False ') + 'AND ' + ('True ' if testInput.y == 1 else 'False ') + '-> ' + ('True ' if guess == 1 else 'False ')
    #
    #     if testInput.label != guess:
    #         strr += "       X"
    #         fail_counter += 1
    #     print(strr)
    # print("--------------------------------------------")
    # print("\nFail counter: %d/%d wrong.\n" % (fail_counter, test_range))
    # print("Learning rate: %f, eras: %d, training inputs: %d\n" % (LR, eras_count, training_inputs_count))

main()

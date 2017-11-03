from graphics import *
from perceptron import *
from input import *
from consts import *
from numpy import interp

""" Variables """
points = []
p = Perceptron(3)
win = GraphWin('Perceptron', WIDTH, HEIGHT, autoflush=False)
training_counter = 0

""" Functions """
def draw_line():
    ps = InputPoint(Point(-1, f(-1)))
    pf = InputPoint(Point(1, f(1)))
    line = Line(Point(ps.interpX(), ps.interpY()), Point(pf.interpX(), pf.interpY()))
    line.setFill('black')
    line.draw(win)

def draw_guessed_line():
    ps = InputPoint(Point(-1, p.guess_y(-1)))
    pf = InputPoint(Point(1, p.guess_y(1)))
    guess_line = Line(Point(ps.interpX(), ps.interpY()), Point(pf.interpX(), pf.interpY()))
    guess_line.setFill('orange')
    guess_line.draw(win)

def mouse_clicked(event):
    global training_counter
    training_counter += 1
    print 'Training no.', training_counter

    win.clear()
    del win.items[:]

    for point in points[:]:
        point.show(win)
        inputs = [point.x, point.y, point.bias]
        target = point.label

        """ Training """
        p.train(inputs, point.label)

        """ Guessing """
        guess = p.guess(inputs)
        circle = Circle(Point(point.interpX(), point.interpY()), 3)
        if guess == target:
            circle.setFill('green')
        else:
            circle.setFill('red')
        circle.draw(win)

    draw_line()
    draw_guessed_line()

"""Creating new random set of points."""
def key_pressed(event):
    win.clear()
    del win.items[:]
    del points[:]

    for i in range(0, POINTS_AMOUNT):
        point = InputPoint(None)
        points.append(point)
        point.show(win)
        target = point.label
        inputs = [point.x, point.y, point.bias]
        circle = Circle(Point(point.interpX(), point.interpY()), 3)
        """Guessing"""
        guess = p.guess(inputs)
        if guess == target:
            circle.setFill('green')
        else:
            circle.setFill('red')

        circle.draw(win)

    draw_line()
    draw_guessed_line()

def main():
    win.setBackground("white")

    for i in range(0, POINTS_AMOUNT):
        point = InputPoint(None)
        points.append(point)
        point.show(win)
    draw_line()

    win.bind('<Button-1>', mouse_clicked)
    win.bind_all('n', key_pressed)
    win.pack()
    win.mainloop()
main()

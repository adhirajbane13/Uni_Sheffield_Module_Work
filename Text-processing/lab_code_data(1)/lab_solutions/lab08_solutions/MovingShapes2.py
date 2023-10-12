
from Shapes import *
from pylab import random as r

####################################################

class MovingShape:
    def __init__(self, frame, shape, diameter):
        self.shape = shape
        self.diameter = diameter
        self.figure = Shape(shape, diameter)

        self.dx = 5 + r() * 10
        if r() < 0.5:
            self.dx *= -1
        self.dy = 5 + r() * 10
        if r() < 0.5:
            self.dy *= -1

        self.calc_min_max_xy(frame)
        
        self.x = self.minx + r() * (self.maxx - self.minx)
        self.y = self.miny + r() * (self.maxy - self.miny)
        self.goto_curr_xy()

    def calc_min_max_xy(self, frame):
        d = self.diameter
        self.minx = d / 2
        self.maxx = frame.width - d / 2
        self.miny = d / 2
        self.maxy = frame.height - d / 2

    def goto_curr_xy(self):
        self.figure.goto(self.x, self.y)

    def moveTick(self):
        self.x += self.dx
        self.y += self.dy
        self.goto_curr_xy()

        if self.x < self.minx or self.x > self.maxx:
            self.dx *= -1
        if self.y < self.miny or self.y > self.maxy:
            self.dy *= -1

    # Provides null interaction behaviour for all cases
    # where first shape is not a square
    def check_collide(self, other):
        pass

####################################################

class Square(MovingShape):
    def __init__(self, frame, diameter):
        MovingShape.__init__(self, frame, 'square', diameter)

    def my_area(self):
        return self.diameter ** 2

    # Provides interaction behaviour for case where first shape is
    # a square - only effects a bounce if second shape also a square
    def check_collide(self, other):
        if other.shape == 'square':
            xdiff = abs(self.x - other.x)
            ydiff = abs(self.y - other.y)
            diam = (self.diameter + other.diameter) / 2.0
            if xdiff < diam and ydiff < diam:
                if xdiff < ydiff:
                    tmp = self.dy
                    self.dy = other.dy
                    other.dy = tmp
                else:
                    tmp = self.dy
                    self.dx = other.dx
                    other.dx = tmp

####################################################

class Diamond(MovingShape):
    def __init__(self, frame, diameter):
        MovingShape.__init__(self, frame, 'diamond', diameter)

    def calc_min_max_xy(self, frame):
        d = self.diameter * 2 ** 0.5
        self.minx = d / 2
        self.maxx = frame.width - d / 2
        self.miny = d / 2
        self.maxy = frame.height - d / 2

    def my_area(self):
        return self.diameter ** 2

####################################################

class Circle(MovingShape):
    def __init__(self, frame, diameter):
        MovingShape.__init__(self, frame, 'circle', diameter)

    def my_area(self):
        return (self.diameter / 2) ** 2 * 3.141

####################################################


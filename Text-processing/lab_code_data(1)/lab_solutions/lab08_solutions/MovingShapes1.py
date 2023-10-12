
from Shapes import *
from pylab import random as r

####################################################

class MovingShape:
    def __init__(self, frame, shape, diameter):
        self.shape = shape
        self.diameter = diameter
        self.figure = Shape(shape, diameter)

        # PART 2: assign random velocities, and 
        # randomly flip some to be negative
        self.dx = 5 + r() * 10
        if r() < 0.5:
            self.dx *= -1
        self.dy = 5 + r() * 10
        if r() < 0.5:
            self.dy *= -1

        # PART 5: calc of min/max x/y values has 
        # been moved out into separate method
        self.calc_min_max_xy(frame)
        
        # PART 3: assign random x/y position, based 
        # on allowable min/max x/y values
        self.x = self.minx + r() * (self.maxx - self.minx)
        self.y = self.miny + r() * (self.maxy - self.miny)
        self.goto_curr_xy()

    # PART 5: separate method calculating min/max x/y 
    # values (so can override this defn in Diamond class)
    def calc_min_max_xy(self, frame):
        d = self.diameter
        self.minx = d / 2
        self.maxx = frame.width - d / 2
        self.miny = d / 2
        self.maxy = frame.height - d / 2

    def goto_curr_xy(self):
        self.figure.goto(self.x, self.y)

    def moveTick(self):
        # PART 1: next three lines implement basic movement
        self.x += self.dx
        self.y += self.dy
        self.goto_curr_xy()

        # PART 4: next two conditionals (i.e. rest of this 
        # method's definition) implement bouncing off walls
        if self.x < self.minx or self.x > self.maxx:
            self.dx *= -1
            self.report_bounce()
        if self.y < self.miny or self.y > self.maxy:
            self.dy *= -1
            self.report_bounce()

    # PART 6: method called whenever a bounce is made, reporting shape and 
    # area - uses area calc methods defined separately in each subclass
    def report_bounce(self):
        print("I'm a bouncing {0}".format(self.shape), end='')
        print(" - my area is {0} sq units!".format(self.my_area()))
        
####################################################

class Square(MovingShape):
    def __init__(self, frame, diameter):
        MovingShape.__init__(self, frame, 'square', diameter)

    # PART 6: area calculation method specific for squares
    def my_area(self):
        return self.diameter ** 2

####################################################

class Diamond(MovingShape):
    def __init__(self, frame, diameter):
        MovingShape.__init__(self, frame, 'diamond', diameter)

    # PART 5: method calculating min/max x/y values - special 
    # definition for Diamond class, overriding definition above
    def calc_min_max_xy(self, frame):
        d = self.diameter * 2 ** 0.5
        self.minx = d / 2
        self.maxx = frame.width - d / 2
        self.miny = d / 2
        self.maxy = frame.height - d / 2

    # PART 6: area calculation method specific for diamonds
    def my_area(self):
        return self.diameter ** 2

####################################################

class Circle(MovingShape):
    def __init__(self, frame, diameter):
        MovingShape.__init__(self, frame, 'circle', diameter)

    # PART 6: area calculation method specific for circles
    def my_area(self):
        return (self.diameter / 2) ** 2 * 3.141

####################################################



from Shapes import *
from pylab import random as r

####################################################

class MovingShape:
    def __init__(self,frame,shape,diameter):
        self.shape = shape
        self.diameter = diameter
        self.figure = Shape(shape,diameter)
        self.x = 0
        self.y = 0

    def goto_curr_xy(self):
        self.figure.goto(self.x,self.y)

    def moveTick(self):
        pass

####################################################

class Square(MovingShape):
    def __init__(self,frame,diameter):
        MovingShape.__init__(self,frame,'square',diameter)

####################################################

class Diamond(MovingShape):
    def __init__(self,frame,diameter):
        MovingShape.__init__(self,frame,'diamond',diameter)

####################################################

class Circle(MovingShape):
    def __init__(self,frame,diameter):
        MovingShape.__init__(self,frame,'circle',diameter)

####################################################


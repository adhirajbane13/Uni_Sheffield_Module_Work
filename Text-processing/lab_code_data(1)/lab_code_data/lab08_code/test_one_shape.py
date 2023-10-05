
from MovingShapes import *

frame = Frame()
shape1 = Square(frame,100)

while not frame.quit:
    shape1.moveTick()

frame.close()


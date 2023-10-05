
from MovingShapes import *

frame = Frame()
numshapes = 3
shapes = []

for n in range(numshapes):
    shapes.append(Square(frame,40))
#    shapes.append(Diamond(frame,40))
#    shapes.append(Circle(frame,40))

while not frame.quit:
    for shape in shapes:
        shape.moveTick()

frame.close()


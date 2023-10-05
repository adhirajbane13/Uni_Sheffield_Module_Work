
from MovingShapes import *

frame = Frame()
numshapes = 2
shapes = []

for n in range(numshapes):
    shapes.append(Square(frame,40))
    shapes.append(Square(frame,30))
    shapes.append(Diamond(frame,40))
    shapes.append(Circle(frame,40))

while not frame.quit:
    for shape in shapes:
        shape.moveTick()

    for i in range(1,len(shapes)):
        for j in range(i):
            shapes[i].check_collide(shapes[j])

frame.close()


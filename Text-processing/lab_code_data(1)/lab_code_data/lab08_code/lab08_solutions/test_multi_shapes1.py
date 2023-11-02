
from MovingShapes1 import Circle, Square, Diamond, Frame

frame = Frame()
numshapes = 2
shapes = []

for n in range(numshapes):
    shapes.append(Square(frame, 40))
    shapes.append(Diamond(frame, 40))
    shapes.append(Circle(frame, 40))

while True:
    frame.graphics_update()
    for shape in shapes:
        shape.move_tick()

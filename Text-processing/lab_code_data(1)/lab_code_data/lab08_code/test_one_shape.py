
from MovingShapes import Circle, Square, Diamond, Frame

frame = Frame()
shape1 = Square(frame, 40)
shape1.goto_curr_xy()

while True:
    frame.graphics_update()
    shape1.move_tick()


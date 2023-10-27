
from Shapes import Shape, Frame

frame = Frame()

s1 = Shape('square', frame, 40)

x = 0
y = 0

while True:
    frame.graphics_update()
    s1.goto(x, y)
    # x += 3
    # y += 1

    

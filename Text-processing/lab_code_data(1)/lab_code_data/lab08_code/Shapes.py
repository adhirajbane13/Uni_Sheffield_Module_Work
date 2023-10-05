
from turtle import Screen, Turtle

####################################################

class Frame:
    def __init__(self,width=800,height=600):
        self.width = width
        self.height = height
        self.screen = Screen()
        self.screen.setup(width=width+100,height=height+100,startx=0,starty=0)
        self.screen.setworldcoordinates(-50,-50,width+50,height+50)
        self.quit = False
        self.draw_frame()

    def close(self):
        self.screen.bye()

    def clear(self):
        self.draw_frame()

    def draw_frame(self):
        self.screen.clear()
        width = self.width
        height = self.height
        boundary = Turtle()
        boundary.hideturtle()
        boundary.speed('fastest')
        boundary.penup()
        boundary.goto(0+2,0-15)
        boundary.write('0')
        boundary.goto(0-8,0)
        boundary.write('0')
        boundary.goto(0,0)
        boundary.pendown()
        boundary.goto(width,0)
        boundary.penup()
        boundary.goto(width-10,0-15)
        boundary.write(str(int(width)))
        boundary.goto(width,0)
        boundary.pendown()
        boundary.goto(width,height)
        boundary.goto(0,height)
        boundary.penup()
        boundary.goto(0-25,height-10)
        boundary.write(str(int(height)))
        boundary.goto(0,height)
        boundary.pendown()
        boundary.goto(0,0)
        boundary.penup()

        self.screen.register_shape("button", ((0,0), (0,85), (25,85), (25,0)))
        t = Turtle(shape="button")
        t.hideturtle()
        t.penup()
        t.fillcolor('pink')
        t.goto(width-95, -20)        
        t.showturtle()

        boundary.goto(width-62, -40)
        boundary.write("Quit",font=("Arial", 12, "normal"))
        self.screen.onclick(self.check_quit)
        
    def check_quit(self,x,y):
        self.quit = 706 < x < 792 and -47 < y < -20

####################################################

class Colours:
    colours = ['red',
               'darkred',
               'blue',
               'darkblue',
               'green',
               'darkgreen',
               'orange',
               'darkorange',
               'brown',
               'turquoise',
               ]

    def get_colour():
        colours = Colours.colours
        colours.append(colours.pop(0))
        return colours[-1]

####################################################

class Shape:
    def __init__(self,shape,diameter):
        self.diameter = diameter
        if shape == 'circle' or shape == 'square':
            t = Turtle(shape=shape)
        elif shape == 'diamond':
            t = Turtle(shape='square')
            t.tilt(45)
        else:
            print('ERROR: shape (%s) not recognised' % shape, file=sys.stderr)
            return
        t.hideturtle()
        t.penup()
        t.color(Colours.get_colour())
        t.fillcolor('')
        t.speed('fastest')
        t.shapesize(diameter/20.0,diameter/20.0,1)
        t.showturtle()
        self.turtle = t

    def goto(self,x,y):
        self.turtle.goto(x,y)

    def vanish(self):
        self.turtle.hideturtle()

####################################################


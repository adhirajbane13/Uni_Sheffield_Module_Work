
from sys import exit, stderr
from graphics import GraphWin, Rectangle, Text, \
    Point, Polygon, Circle, update

####################################################

class Frame:
    def __init__(self, width = 800, height = 600, title = ''):
        self.width = width
        self.height = height
        self.paused = False
        self.title = title
        self.screen = self.draw_screen()
        self.quit_button = QuitButton(self)
        self.step_button = StepButton(self)
        self.pause_button = PauseButton(self)
        
    def draw_screen(self):
        screen = GraphWin(self.title, self.width+100, self.height+100, autoflush=False)
        screen.setCoords(-50, -50, self.width+50, self.height+50)
        border = Rectangle(Point(0,0), Point(self.width, self.height))
        border.draw(screen)
        Text(Point(-6, 4), 0).draw(screen)
        Text(Point(4, -9), 0).draw(screen)
        Text(Point(self.width-8, -9), self.width).draw(screen)
        Text(Point(-12,self.height-8), self.height).draw(screen)
        return screen

    def close(self):
        self.screen.close()

    def graphics_update(self):
        update(40)
        point = self.screen.checkMouse()
        self.check_buttons(point)

    def check_buttons(self, point):
        self.quit_button.check_action(point)
        paused = self.paused    # record pre-click state of paused, 
                                # as might get toggled
        step_clicked = self.step_button.check_action(point)
        pause_clicked = self.pause_button.check_action(point)
        if ((not paused and pause_clicked) or 
            (paused and not pause_clicked and not step_clicked)):
            point = self.screen.getMouse()
            self.check_buttons(point)

####################################################

class Button:
    def __init__(self, frame, button_text, center, width=80, height=30, 
                 color=['lightgray']):
        x, y = center.getX(), center.getY()
        self.top = y + height / 2
        self.bottom = self.top - height
        self.left = x - width / 2
        self.right = self.left + width
        self.frame = frame
        button = Rectangle(Point(self.left, self.top), 
                           Point(self.right, self.bottom))
        if type(color) == type(''):
            color = [color]
        self.color = color
        button.setFill(color[0])
        button.draw(frame.screen)
        button_text = Text(center, button_text)
        button_text.setSize(18)
        button_text.draw(frame.screen)
        self.button = button
        
    def color_cycle(self):
        self.color.append(self.color.pop(0))
        self.button.setFill(self.color[0])

    def clicked(self, point):
        return (point and 
                self.left < point.getX() < self.right and
                self.bottom < point.getY() < self.top)

    def check_action(self, point):
        if self.clicked(point):
            return self.action()

    def action(self):
        raise Exception('Method "action" not defined for subclass')

####################################################

class StepButton(Button):
    def __init__(self, frame):
        Button.__init__(self, frame, "Step", Point(frame.width - 260, -24),
                        color=['burlywood', 'darkseagreen'])
        self.active = False

    def action(self):
        return self.active

    def toggle(self):
        self.active = not self.active
        self.color_cycle()

####################################################

class QuitButton(Button):
    def __init__(self, frame):
        Button.__init__(self, frame, "Quit", Point(frame.width - 80, -24))

    def action(self):
        self.frame.close()
        exit(0)

####################################################

class PauseButton(Button):
    def __init__(self, frame):
        Button.__init__(self, frame, "Pause", 
                        Point(frame.width - 170, -24),
                        color=['aquamarine', 'salmon'])
        self.paused = False

    def action(self):
        self.color_cycle()
        self.frame.step_button.toggle()
        self.frame.paused = not self.frame.paused
        return True

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
    def __init__(self, shape, frame, diameter):
        self.frame = frame
        self.figure = None
        self.undrawn = True
        self.x = 0
        self.y = 0
        if shape == 'circle':
            self.figure = Circle(Point(0,0), diameter/2)
        elif shape == 'square':
            d = diameter/2
            self.figure = Rectangle(Point(d,d), Point(-d,-d))
        elif shape == 'diamond':
            d = diameter / (2 ** 0.5)
            self.figure = Polygon(Point(d,0), Point(0,d), Point(-d,0), Point(0,-d))
            pass
        else:
            print('ERROR: shape (%s) not recognised' % shape, file = stderr)
            return
        self.figure.setOutline(Colours.get_colour())

    def goto(self, x, y):
        self.figure.move(x - self.x, y - self.y)
        self.x = x
        self.y = y
        if self.undrawn:
            self.figure.draw(self.frame.screen)
            self.undrawn = False

    def vanish(self):
        self.figure.undraw()

####################################################


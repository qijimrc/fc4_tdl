from environment import Connect4

from typing import List, Dict
import turtle

class Connect4Render(Connect4):

    def __init__(self, n_row:int=6, n_col:int=7, obs_0: List=None, unit: int=100):
        super().__init__(n_row, n_col, obs_0)
        # unit size
        self.t = None
        self.unit = 100


    def draw_box(self, x, y, piece=None, fillcolor='', line_color='gray'):
        self.t.hideturtle()
        self.t.up()
        self.t.goto(x * self.unit, y * self.unit)
        self.t.color(line_color)
        self.t.fillcolor(fillcolor)
        self.t.setheading(90)
        self.t.down()
        self.t.begin_fill()
        for _ in range(4):
            self.t.forward(self.unit)
            self.t.right(90)
        self.t.end_fill()
        # draw piece
        if piece:
            self.t.up()
            self.t.goto(x*self.unit+self.unit*0.8, y*self.unit+self.unit/2)
            self.t.down()
            self.t.fillcolor(piece)
            self.t.begin_fill()
            self.t.circle(self.unit/3)
            self.t.end_fill()
            self.t.up()

    def move_player(self, x, y, piece, value=None):
        self.t.hideturtle()
        self.t.up()
        self.t.setheading(90)
        # self.t.fillcolor('red')
        if value:
            self.t.hideturtle()
            self.t.up()
            self.t.goto(x*self.unit+self.unit*0.8, y*self.unit+self.unit/2)
            self.t.down()
            self.t.fillcolor(piece)
            self.t.begin_fill()
            self.t.circle(self.unit/3)
            self.t.end_fill()
            self.t.up()
            value = round(value, 16)
            self.t.goto((x + 0.5) * self.unit, (y + 0.5) * self.unit)
            self.t.fillcolor('red')
            # self.t.showturtle()
            self.t.write(value, align="center", font=("Courier", 10))
        else:
            self.t.up()
            self.t.goto(x*self.unit+self.unit*0.8, y*self.unit+self.unit/2)
            self.t.down()
            self.t.fillcolor(piece)
            self.t.begin_fill()
            self.t.circle(self.unit/3)
            self.t.end_fill()
            self.t.up()


    def render(self, act:Dict=None, show_value: bool=False, reset: bool=False):
        if self.t == None or reset:
            self.t = turtle.Turtle()
            self.t.hideturtle()
            self.wn = turtle.Screen()
            self.wn.setup(self.unit * self.n_col + 100,
                          self.unit * self.n_row + 100)
            self.wn.setworldcoordinates(0, 0, self.unit * self.n_col,
                                        self.unit * self.n_row)
            self.t.shape('circle')
            self.t.width(2)
            self.t.speed(0)
            self.t.color('black')
            for i in range(self.n_row):
                for j in range(self.n_col):
                    k = self.n_row - i -1
                    x, y = j, i
                    if self.board[k][j] == 0:   # empty and not reachable
                        self.draw_box(x, y, None)
                    elif self.board[k][j] == 1:   #  agent1
                        self.draw_box(x, y, 'yellow')
                    elif self.board[k][j] == 2:   #  agent2
                        self.draw_box(x, y, 'red')
                    elif self.board[k][j] == 3:   #  agent1
                        self.draw_box(x, y, None)
                    else:
                        raise EnvironmentError
            self.t.shape("circle")

        if act:
            # x_pos_ori, y_pos_ori = self.t.pos()
            occ_row, occ_col = act["occupation"]
            value = act["value"]
            x = occ_col
            y = self.n_row - occ_row -1
            if act["player"] == 1:
                piece = "yellow"
            else:
                piece = "red"
            if not show_value: value = None
            
            self.move_player(x, y, piece, value)







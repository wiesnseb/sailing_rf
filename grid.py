import numpy as np
import time
import tkinter as tk

UNIT = 40   # pixels per cell (width and height)
GRID_H = 10  # height of the entire grid in cells
GRID_W = 10  # width of the entire grid in cells
origin = np.array([UNIT/2, UNIT/2])


class Grid(tk.Tk, object):
    def __init__(self, start_port, end_port, islands=[],rough_waters=[]):
        super(Grid, self).__init__()
        self.action_space = ['u', 'd', 'l', 'r']
        self.n_actions = len(self.action_space)
        self.wallblocks = []
        self.pitblocks=[]
        # pixels per cell (width and height)
        self.UNIT = 40
        # height of the entire grid in cells
        self.GRID_H = 10
        # width of the entire grid in cells
        self.GRID_W = 10
        self.title('Ocean')
        self.geometry('{0}x{1}'.format(GRID_H * UNIT, GRID_W * UNIT))
        self.build_shape_grid(start_port, end_port, islands, rough_waters)

    def build_shape_grid(self,start_port,end_port, islands,rough_waters):
        self.canvas = tk.Canvas(self, bg='lightblue',
                           height=GRID_H * UNIT,
                           width=GRID_W * UNIT)

        # create grids
        for c in range(0, GRID_W * UNIT, UNIT):
            x0, y0, x1, y1 = c, 0, c, GRID_H * UNIT
            self.canvas.create_line(x0, y0, x1, y1)
        for r in range(0, GRID_H * UNIT, UNIT):
            x0, y0, x1, y1 = 0, r, GRID_W * UNIT, r
            self.canvas.create_line(x0, y0, x1, y1)


        for x,y in islands:
            self.add_wall(x,y)
        for x,y in rough_waters:
            self.add_pit(x,y)
        self.add_goal(end_port[0],end_port[1])
        self.add_agent(start_port[0],start_port[1])
        self.canvas.pack()

    # Add a solid wall block at coordinate for centre of bloc
    def add_wall(self, x, y):
        wall_center = origin + np.array([UNIT * x, UNIT*y])
        self.wallblocks.append(self.canvas.create_rectangle(
            wall_center[0] - 15, wall_center[1] - 15,
            wall_center[0] + 15, wall_center[1] + 15,
            fill='bisque4'))

    # Add a solid pit block at coordinate for centre of bloc
    def add_pit(self, x, y):
        pit_center = origin + np.array([UNIT * x, UNIT*y])
        self.pitblocks.append(self.canvas.create_rectangle(
            pit_center[0] - 15, pit_center[1] - 15,
            pit_center[0] + 15, pit_center[1] + 15,
            fill='aquamarine4'))

    # Add a solid goal for goal at coordinate for centre of bloc
    def add_goal(self, x=4, y=4):
        goal_center = origin + np.array([UNIT * x, UNIT*y])

        self.goal = self.canvas.create_oval(
            goal_center[0] - 15, goal_center[1] - 15,
            goal_center[0] + 15, goal_center[1] + 15,
            fill='yellow')

    # Add a solid wall red block for agent at coordinate for centre of block
    def add_agent(self, x=0, y=0):
        agent_center = origin + np.array([UNIT * x, UNIT*y])

        self.agent = self.canvas.create_rectangle(
            agent_center[0] - 15, agent_center[1] - 15,
            agent_center[0] + 15, agent_center[1] + 15,
            fill='red')

    def reset(self, value = 1, resetAgent=True):
        self.update()
        time.sleep(0.2)
        if(value == 0):
            return self.canvas.coords(self.agent)
        else:
            # Reset Agent
            if(resetAgent):
                self.canvas.delete(self.agent)
                self.agent = self.canvas.create_rectangle(origin[0] - 15, origin[1] - 15,
                origin[0] + 15, origin[1] + 15,
                fill='red')

            return self.canvas.coords(self.agent)

    # computeReward - definition of reward function
    def computeReward(self, currstate, action, nextstate):
            reverse=False
            if nextstate == self.canvas.coords(self.goal):
                reward = 1
                done = True
                nextstate = 'terminal'
            #elif nextstate in [self.canvas.coords(self.pit1), self.canvas.coords(self.pit2)]:
            elif nextstate in [self.canvas.coords(w) for w in self.wallblocks]:
                reward = -0.3
                done = False
                nextstate = currstate
                reverse=True
                #print("Wall penalty:{}".format(reward))
            elif nextstate in [self.canvas.coords(w) for w in self.pitblocks]:
                reward = -10
                done = True
                nextstate = 'terminal'
                reverse=False
                #print("Wall penalty:{}".format(reward))
            else:
                reward = -0.1
                done = False
            return reward,done, reverse

    #step - definition of one-step dynamics function
    def step(self, action):
        s = self.canvas.coords(self.agent)
        base_action = np.array([0, 0])
        # up
        if action == 0:
            if s[1] > UNIT:
                base_action[1] -= UNIT
        # down
        elif action == 1:
            if s[1] < (GRID_H - 1) * UNIT:
                base_action[1] += UNIT
        # right
        elif action == 2:
            if s[0] < (GRID_W - 1) * UNIT:
                base_action[0] += UNIT
        # left
        elif action == 3:
            if s[0] > UNIT:
                base_action[0] -= UNIT

        self.canvas.move(self.agent, base_action[0], base_action[1])  # move agent

        s_ = self.canvas.coords(self.agent)  # next state
        #print("s_.coords:{}({})".format(self.canvas.coords(self.agent),type(self.canvas.coords(self.agent))))
        #print("s_:{}({})".format(s_, type(s_)))

        # call the reward function
        reward, done, reverse = self.computeReward(s, action, s_)
        if(reverse):
            # move agent back
            self.canvas.move(self.agent, -base_action[0], -base_action[1])
            s_ = self.canvas.coords(self.agent)  

        return s_, reward, done

    def render(self, sim_speed=.01):
        time.sleep(sim_speed)
        self.update()


def update():
    for t in range(10):
        print("The value of t is", t)
        s = env.reset()
        while True:
            env.render()
            a = 1
            s, r, done = env.step(a)
            if done:
                break


if __name__ == '__main__':
    env = Grid()
    env.after(100, update)
    env.mainloop()
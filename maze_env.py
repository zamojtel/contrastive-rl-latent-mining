
import numpy as np


class MazeEnv:
    def __init__(self):
        self.state = None
        self.goal = None

        self.action_map = {
            0: [-1,0], # up
            1: [1,0],  # down
            2: [0,-1], # left
            3: [0,1],  # right
        }

        self.maze_4 = np.array(
            [
                [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [3, 0, 1, 4, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
                [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            ]
        )

    def reset(self):
        start = np.where(self.maze_4 == 3)
        finish = np.where(self.maze_4 == 4)
        start_row = start[0][0]
        start_col = start[1][0]
        end_row = finish[0][0]
        end_col = finish[1][0]

        self.state = (start_row,start_col)
        self.goal = (end_row,end_col)

        return {
            'observation':self.state,
            'desired_goal':self.goal
        }
        
    def step(self,action):
        current_move = self.action_map[action]
        new_row = self.state[0]+current_move[0]
        new_col = self.state[1]+current_move[1]
        



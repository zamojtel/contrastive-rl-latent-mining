import numpy as np

class MazeEnv:
    def __init__(self,maze):
        self.state = None
        self.goal = None
        self.maze = maze
        self.n_rows = len(maze)
        self.n_cols = len(maze[0])
        self.is_done = False

        self.action_map = {
            0: [-1,0], # up
            1: [1,0],  # down
            2: [0,-1], # left
            3: [0,1],  # right
        }


    def reset(self):
        start = np.where(self.maze == 3)
        finish = np.where(self.maze == 4)
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

        if new_row >= 0 and new_row < self.n_rows and new_col >= 0 and new_col < self.n_cols:
            
            if self.maze[new_row][new_col] != 1:
                self.state = (new_row,new_col)
            else:
                pass
        
        is_done = (self.state == self.goal)
        reward = 0 if is_done else -1

        return (
            {
            'observation':self.state,
            'desired_goal':self.goal
            },
            reward,
            is_done,
            {}
        )
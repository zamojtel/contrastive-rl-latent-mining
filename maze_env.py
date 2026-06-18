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
        
    def step(self,state,action):
        current_move = self.action_map[action]
        new_row = state[0]+current_move[0]
        new_col = state[1]+current_move[1]
        # np.where()
        is_valid_move =(
            ( new_row >= 0 ) & ( new_row < self.n_rows ) &
            ( new_col >= 0 ) & ( new_col < self.n_cols ) &
            (self.maze[new_row,new_col] != 1 )
        )
        # if new_row >= 0 and new_row < self.n_rows and new_col >= 0 and new_col < self.n_cols:
            
        #     if self.maze[new_row][new_col] != 1:
        #         state = (new_row,new_col)
        #     else:
        #         pass
        final_row = np.where(is_valid_move,new_row,state[0])
        final_col = np.where(is_valid_move,new_col,state[1])

        next_state = (final_row,final_col)
        is_done = (next_state == self.goal)
        reward = 0 if is_done else -1

        return {
            'obs':{
                'observation': state,
                'desired_goal':self.goal,
            },
            'reward': reward,
            'done': is_done,
            'info': {
                'row': state[0],
                'col': state[1],
            }
        }
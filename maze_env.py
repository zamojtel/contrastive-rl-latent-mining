
import jax.numpy as jnp
from brax.envs import State

class MazeEnv:
    def __init__(self, maze):
        self.maze = maze
        self.n_rows = len(maze)
        self.n_cols = len(maze[0])
        self.action_size = 4 
        self.state_dim = 2
        self.goal_indices = [0,1]
        self.observation_size = 4 # bo 2 na current_pos i 2 na desired_goal

        start = jnp.where(self.maze == 3)
        finish = jnp.where(self.maze == 4)
        
        self.start_pos = (start[0][0], start[1][0]) 
        self.goal = (finish[0][0], finish[1][0])
        
        self.action_map = jnp.array([
            [-1,0], # up
            [1,0],  # down
            [0,-1], # left
            [0,1],  # right
        ])

    def reset(self, rng):
        return State(
            pipeline_state=None,
            obs = jnp.array(
                self.start_pos+self.goal
            ),
            reward=jnp.zeros(()), 
            done=jnp.zeros(()),
            metrics={
                'success': jnp.zeros(()),
                'success_easy': jnp.zeros(()),
                'dist': jnp.zeros(()),
                'distance_from_origin': jnp.zeros(()),
            },
            info={}
        )
        
    def step(self,state,action):
        # przeliczanie wektorowe
        action_idx = jnp.argmax(action)
        current_move = self.action_map[action_idx]
        current_row = state.obs[0]
        current_col = state.obs[1]
        new_row = current_row + current_move[0]
        new_col = current_col + current_move[1]

        is_valid_move =(
            ( new_row >= 0 ) & ( new_row < self.n_rows ) &
            ( new_col >= 0 ) & ( new_col < self.n_cols ) &
            (self.maze[new_row,new_col] != 1 )
        )

        final_row = jnp.where(is_valid_move,new_row,current_row)
        final_col = jnp.where(is_valid_move,new_col,current_col)
        next_state = (final_row,final_col)
        
        dist = jnp.abs(final_row-self.goal[0]) + jnp.abs(final_col-self.goal[1])
        distance_from_origin = jnp.abs(final_row-self.start_pos[0]) + jnp.abs(final_col-self.start_pos[1])

        is_done = ((final_row == self.goal[0]) & (final_col == self.goal[1]))

        reward = jnp.where(is_done,0.0,-1.0)
        
        return State(
            pipeline_state=None,
            obs = jnp.array(
                next_state+self.goal
            ),
            reward = reward,
            done = is_done.astype(jnp.float32),
            metrics = {**state.metrics,'success': is_done.astype(jnp.float32),
            'success_easy': is_done.astype(jnp.float32),  
            'dist': dist.astype(jnp.float32),
            'distance_from_origin': distance_from_origin.astype(jnp.float32),
            },
            info = state.info,
        )
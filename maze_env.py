
import jax.numpy as jnp
from brax.envs import State

class MazeEnv:
    def __init__(self, maze):
        self.maze = maze
        self.num_mazes, self.n_rows, self.n_cols = maze.shape 
        self.action_size = 4 
        self.state_dim = 2
        self.goal_indices = [0,1]
        self.observation_size = 4

        starts = jnp.where(self.maze == 3)
        finishes = jnp.where(self.maze == 4)

        self.start_positions = jnp.stack([starts[1], starts[2]], axis=-1)
        self.goal_positions = jnp.stack([finishes[1], finishes[2]], axis=-1)
        
        self.action_map = jnp.array([
            [-1,0], # up
            [1,0],  # down
            [0,-1], # left
            [0,1],  # right
        ])

    def reset(self, rng):
        maze_id = jax.random.randint(rng, shape=(), minval=0, maxval=4)
        current_start = self.start_positions[maze_id]
        current_goal = self.goal_positions[maze_id]
        obs = jnp.concatenate([current_start,current_goal])
        return State(
            pipeline_state=None,
            obs = obs,
            reward=jnp.zeros(()), 
            done=jnp.zeros(()),
            metrics={
                'success': jnp.zeros(()),
                'success_easy': jnp.zeros(()),
                'dist': jnp.zeros(()),
                'distance_from_origin': jnp.zeros(()),
            },
            info={'maze_index': maze_id}
        )
        
    def step(self,state,action):
        current_maze_id = state.info['maze_index']
        action_idx = jnp.argmax(action)
        current_move = self.action_map[action_idx]
        current_row = state.obs[0]
        current_col = state.obs[1]
        new_row = current_row + current_move[0]
        new_col = current_col + current_move[1]

        is_valid_move =(
            ( new_row >= 0 ) & ( new_row < self.n_rows ) &
            ( new_col >= 0 ) & ( new_col < self.n_cols ) &
            (self.maze[current_maze_id,new_row,new_col] != 1 )
        )

        final_row = jnp.where(is_valid_move,new_row,current_row)
        final_col = jnp.where(is_valid_move,new_col,current_col)
        next_state = (final_row,final_col)

        current_start = self.start_positions[current_maze_id]
        current_goal = self.goal_positions[current_maze_id]
        dist = jnp.abs(final_row-current_goal[0]) + jnp.abs(final_col-current_goal[1])
        distance_from_origin = jnp.abs(final_row-current_start[0]) + jnp.abs(final_col-current_start[1])

        is_done = ((final_row == current_goal[0]) & (final_col == current_goal[1]))

        reward = jnp.where(is_done,0.0,-1.0)
        current_pos = jnp.array([final_row,final_col])
        obs = jnp.concatenate([current_pos,current_goal])

        return State(
            pipeline_state=None,
            obs = obs,
            reward = reward,
            done = is_done.astype(jnp.float32),
            metrics = {**state.metrics,'success': is_done.astype(jnp.float32),
            'success_easy': is_done.astype(jnp.float32),  
            'dist': dist.astype(jnp.float32),
            'distance_from_origin': distance_from_origin.astype(jnp.float32),
            },
            info = state.info,
        )
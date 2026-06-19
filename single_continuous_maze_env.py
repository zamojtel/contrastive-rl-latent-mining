import jax
import jax.numpy as jnp
from brax.envs import State

class SingleContinuousMazeEnv:
    def __init__(self, maze_2d):
        self.maze = jnp.array(maze_2d)
        self.n_rows, self.n_cols = self.maze.shape 
        
        self.action_size = 2
        self.observation_size = 4

        free_cells = jnp.column_stack(jnp.where(self.maze != 1))
        self.free_cells = free_cells.astype(jnp.float32)
        self.num_free_cells = len(free_cells)

    def reset(self, rng):
        rng1, rng2, rng3, rng4 = jax.random.split(rng, 4)
    
        start_idx = jax.random.randint(rng1, shape=(), minval=0, maxval=self.num_free_cells)
        goal_idx = jax.random.randint(rng2, shape=(), minval=0, maxval=self.num_free_cells)
        
        start_pos = self.free_cells[start_idx] + jax.random.uniform(rng3, shape=(2,), minval=0.1, maxval=0.9)
        goal_pos = self.free_cells[goal_idx] + jax.random.uniform(rng4, shape=(2,), minval=0.1, maxval=0.9)

        norm_factor = jnp.array([self.n_rows, self.n_cols], dtype=jnp.float32)
        obs = jnp.concatenate([start_pos / norm_factor, goal_pos / norm_factor])
        
        return State(
            pipeline_state=start_pos,
            obs=obs,
            reward=jnp.zeros(()), 
            done=jnp.zeros(()),
            metrics={'success': jnp.zeros(()), 'dist': jnp.zeros(())},
            info={'goal': goal_pos}
        )
        
    def step(self, state, action):
        current_pos = state.pipeline_state
        current_goal = state.info['goal']
        
        action = jnp.clip(action, -1.0, 1.0)
        new_pos = current_pos + action * 0.5
        new_pos = jnp.clip(new_pos, 0.0, jnp.array([self.n_rows - 0.01, self.n_cols - 0.01]))
        
        grid_pos = jnp.floor(new_pos).astype(jnp.int32)
        is_wall = self.maze[grid_pos[0], grid_pos[1]] == 1
        
        final_pos = jnp.where(is_wall, current_pos, new_pos)
        
        dist = jnp.linalg.norm(final_pos - current_goal)
        is_done = dist < 0.5
        
        reward = jnp.where(is_done, 0.0, -1.0)
        
        norm_factor = jnp.array([self.n_rows, self.n_cols], dtype=jnp.float32)
        obs = jnp.concatenate([final_pos / norm_factor, current_goal / norm_factor])

        return State(
            pipeline_state=final_pos,
            obs=obs,
            reward=reward,
            done=is_done.astype(jnp.float32),
            metrics={**state.metrics, 'success': is_done.astype(jnp.float32), 'dist': dist.astype(jnp.float32)},
            info=state.info,
        )
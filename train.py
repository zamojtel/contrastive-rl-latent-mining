from maze_env import MazeEnv
import numpy as np
from jaxgcrl.agents import CRL


maze = np.array(
        [
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [3, 0, 1, 4, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        ]
    )

env = MazeEnv(maze)
agent = CRL()

for episode in range(10_000):
    obs = env.reset()
    is_done = False
    steps = 0
    while not is_done :
        steps+=1
        flat_obs = np.array(obs['observation']+obs['desired_goal'])
        action = agent.
        obs,reward,is_done , _ = env.step(action)
        next_flat_obs = np.array(obs['observation']+obs['desired_goal'])
        agent.update(flat_obs,action,reward,next_flat_obs,is_done)
    print(f"Number of steps {steps}")

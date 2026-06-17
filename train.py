from maze_env import MazeEnv
import numpy as np

maze =np.array(
        [
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [3, 0, 1, 4, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        ]
    )

env = MazeEnv(maze)
obs = env.reset()
is_done = False
steps = 0
# random walk
while not is_done :
    steps+=1
    flat_obs = np.array(obs['observation']+obs['desired_goal'])
    action = np.random.randint(0,4)
    obs,_,is_done , _ = env.step(action)


print(f"Number of steps {steps}")

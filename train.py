from maze_env import MazeEnv
import jax.numpy as jnp
from jaxgcrl.agents import CRL
import time
import os


_orig_perf = time.perf_counter
_orig_time = time.time
_tick = [0.0]

def fake_time():
    _tick[0] += 1e-7
    return _orig_perf() + _tick[0]

time.perf_counter = fake_time
time.time = fake_time

def print_progress(num_steps, metrics,*args, **kwargs):
    print(f"Steps: {num_steps} | Success: {metrics['eval/episode_success']:.2f} | Dist to Goal: {metrics['eval/episode_dist']:.2f}")

class Config:
    def __init__(self):
        self.num_envs = 1024
        self.num_eval_envs = 256
        self.episode_length = 257
        self.action_repeat = 1
        self.total_env_steps = 50_000_000
        self.num_evals = 10
        self.seed = 42
        self.visualization_interval = 10
        self.checkpoint_logdir = "./my_agent"

maze = jnp.array([
# Labirynt 0
[
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
[3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4],
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
],
# Labirynt 1
[
[0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
[0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
[3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4],
[0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
[0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
],
# Labirynt 2
[
[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
[0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0],
[3, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 4],
[0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0],
[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
],
# Labirynt 3
[
[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
[0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
[3, 0, 1, 4, 0, 0, 0, 0, 0, 0, 0, 0],
[0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
]
])

env = MazeEnv(maze)
agent = CRL()
print("Batch size agenta to:", agent.batch_size)
print("starting compilation and training")
my_config = Config()
os.makedirs("./my_agent", exist_ok=True)
agent.train_fn(
    config = my_config,
    train_env = env,
    eval_env = env,
    progress_fn=print_progress,
)

print("finished")

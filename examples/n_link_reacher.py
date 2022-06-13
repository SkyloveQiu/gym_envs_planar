# pylint: disable=import-outside-toplevel
from cmath import e
from random import uniform
from tabnanny import check
import gym
from matplotlib.pyplot import step
from torch import rand
import planarenvs.n_link_reacher  # pylint: disable=unused-import
import numpy as np
import os

from stable_baselines3 import DDPG, SAC
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.callbacks import BaseCallback
from goal import staticGoal
obstacles = False
goal = True


class CheckpointCallback(BaseCallback):
    """
    Callback for saving a model every ``save_freq`` calls
    to ``env.step()``.

    .. warning::

      When using multiple environments, each call to  ``env.step()``
      will effectively correspond to ``n_envs`` steps.
      To account for that, you can use ``save_freq = max(save_freq // n_envs, 1)``

    :param save_freq:
    :param save_path: Path to the folder where the model will be saved.
    :param name_prefix: Common prefix to the saved models
    :param verbose:
    """

    def __init__(self, save_freq: int, save_path: str, name_prefix: str = "rl_model", verbose: int = 0):
        super(CheckpointCallback, self).__init__(verbose)
        self.save_freq = save_freq
        self.save_path = save_path
        self.name_prefix = name_prefix

    def _init_callback(self) -> None:
        # Create folder if needed
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.save_freq == 0:
            staticGoal.shuffle()
            path = os.path.join(self.save_path, f"{self.name_prefix}_{self.num_timesteps}_steps")
            self.model.save(path)
            if self.verbose > 1:
                print(f"Saving model checkpoint to {path}")
        return True

def make_env(env_id, rank, seed=0):
    """
    Utility function for multiprocessed env.

    :param env_id: (str) the environment ID
    :param num_env: (int) the number of environments you wish to have in subprocesses
    :param seed: (int) the inital seed for RNG
    :param rank: (int) index of the subprocess
    """
    def _init():
        env = gym.make(env_id)
        env.seed(seed + rank)
        sensors = True
        if sensors:
            from planarenvs.sensors.goal_sensor import (
                GoalSensor,
            )
            # from planarenvs.sensors.obstacle_sensor import (
            #     ObstacleSensor,
            # )
            # obst_sensor_pos = ObstacleSensor(nb_obstacles=2, mode="position")
            # env.add_sensor(obst_sensor_pos)
            # obst_sensor_dist = ObstacleSensor(nb_obstacles=2, mode="distance")
            # env.add_sensor(obst_sensor_dist)
            goal_dist_observer = GoalSensor(nb_goals=1, mode="distance")
            env.add_sensor(goal_dist_observer)
            goal_pos_observer = GoalSensor(nb_goals=1, mode="position")
            env.add_sensor(goal_pos_observer)
        
        if obstacles:
            from examples.obstacles import (
                sphereObst1,
                sphereObst2,
                dynamicSphereObst2,
            )

            env.add_obstacle(sphereObst1)
            env.add_obstacle(sphereObst2)
            env.add_obstacle(dynamicSphereObst2)
        if goal:
            env.add_goal(staticGoal)
        return env
    set_random_seed(seed)
    return _init

def run_n_link_reacher(
    n_steps=1000,
    render: bool = True,
    goal: bool = False,
    obstacles: bool = False,
):
    """
    Minimal example for n-link planar robot arm.

    The n-link-arm is a n-degrees of freedom robotic arm operating in the
    two-dimensional plane. In a sense, it is extended pendulum. The observation
    is the state of the joints:
        x: [`q`]
        xdot: [`qdot`]
    """
    env_id = "nLink-reacher-vel-v0"
    n = 2
    num_cpu = 20
    env = SubprocVecEnv([make_env(env_id, i) for i in range(num_cpu)])
    # sensors = True
    # if sensors:
    #     from planarenvs.sensors.goal_sensor import (
    #         GoalSensor,
    #     )
    #     # from planarenvs.sensors.obstacle_sensor import (
    #     #     ObstacleSensor,
    #     # )
    #     # obst_sensor_pos = ObstacleSensor(nb_obstacles=2, mode="position")
    #     # env.add_sensor(obst_sensor_pos)
    #     # obst_sensor_dist = ObstacleSensor(nb_obstacles=2, mode="distance")
    #     # env.add_sensor(obst_sensor_dist)
    #     goal_dist_observer = GoalSensor(nb_goals=1, mode="distance")
    #     env.add_sensor(goal_dist_observer)
    #     goal_pos_observer = GoalSensor(nb_goals=1, mode="position")
    #     env.add_sensor(goal_pos_observer)
    # if obstacles:
    #     from examples.obstacles import (
    #         sphereObst1,
    #         sphereObst2,
    #         dynamicSphereObst2,
    #     )

    #     env.add_obstacle(sphereObst1)
    #     env.add_obstacle(sphereObst2)
    #     env.add_obstacle(dynamicSphereObst2)
    # if goal:
    #     from examples.goal import (
    #         splineGoal,
    #     )
    #     env.add_goal(splineGoal)
    checkpoint_callback = CheckpointCallback(save_freq=5000, save_path='./logs/',
                                         name_prefix='rl_model')
    model = SAC("MultiInputPolicy", env, verbose=1,learning_rate=0.001)
    model.learn(total_timesteps=1000000, log_interval=10,callback=checkpoint_callback)
    model.save("SAC")
    model = SAC.load("SAC")
    action = np.ones(n) * 8 * 0.01
    n_steps = 100000
    obs = env.reset()
    print("Starting episode")
    for i in range(n_steps):
        action, _states = model.predict(obs,deterministic=True)
        obs, _, _, _ = env.step(action)


def simulate_n_link_reacher():
    env = gym.make("nLink-reacher-vel-v0", render = True, n=2,dt = 0.01)
    from planarenvs.sensors.goal_sensor import (
                GoalSensor,
            )
    goal_dist_observer = GoalSensor(nb_goals=1, mode="distance")
    env.add_sensor(goal_dist_observer)
    goal_pos_observer = GoalSensor(nb_goals=1, mode="position")
    env.add_sensor(goal_pos_observer)
    env.add_goal(staticGoal)
    model = SAC.load("SAC")
    obs = env.reset()
    n_steps = 10000
    for i in range(n_steps):
        action, _state = model.predict(obs,deterministic=True)
        obs, _, _, _ = env.step(action)

if __name__ == "__main__":
    obstacles = True
    goal = True
    simulate_n_link_reacher()

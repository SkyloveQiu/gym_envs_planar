# pylint: disable=import-outside-toplevel
from cmath import e
import gym
import planarenvs.n_link_reacher  # pylint: disable=unused-import
import numpy as np

from stable_baselines3 import DDPG, SAC
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.utils import set_random_seed
obstacles = False
goal = True

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
            from examples.goal import (
                splineGoal,
            )
            env.add_goal(splineGoal)
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
    num_cpu = 200
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
    model = SAC("MultiInputPolicy", env, verbose=1,learning_rate=0.001)
    model.learn(total_timesteps=10000000, log_interval=10)
    model.save("SAC")
    model = SAC.load("SAC")
    action = np.ones(n) * 8 * 0.01
    n_steps = 100000
    obs = env.reset()
    print("Starting episode")
    for i in range(n_steps):
        action, _states = model.predict(obs,deterministic=True)
        obs, _, _, _ = env.step(action)


if __name__ == "__main__":
    obstacles = True
    goal = True
    run_n_link_reacher(goal=goal, obstacles=obstacles)

# pylint: disable=import-outside-toplevel
import gym
import planarenvs.n_link_reacher  # pylint: disable=unused-import
import numpy as np

<<<<<<< HEAD
from stable_baselines3 import DDPG, SAC
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
obstacles = False
goal = True
=======
>>>>>>> ff805d4 (Fixes dynamic goal issues. Formats examples. Adds testing of examples. (#47))

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
<<<<<<< HEAD
    n = 2
    env = gym.make("nLink-reacher-vel-v0", render=True, n=n, dt=0.01)
=======
    n = 3
    env = gym.make("nLink-reacher-vel-v0", render=render, n=n, dt=0.01)
    action = np.ones(n) * 8 * 0.01
    ob = env.reset(pos=np.random.rand(n))
>>>>>>> ff805d4 (Fixes dynamic goal issues. Formats examples. Adds testing of examples. (#47))
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
            staticGoal,
        )
<<<<<<< HEAD
        env.add_goal(staticGoal)
    n_actions = env.action_space.shape[-1]
    action_noise = NormalActionNoise(mean=0.1 * np.ones(n_actions), sigma=0.2 * np.ones(n_actions))
    model = SAC("MultiInputPolicy", env, verbose=1,learning_rate=0.001)
    model.learn(total_timesteps=100000, log_interval=10)
    model.save("ddpg")
    env = model.get_env()

    action = np.ones(n) * 8 * 0.01
    n_steps = 100000
    ob = env.reset()
    if goal:
        from examples.goal import (
            staticGoal,
        )
        env.add_goal(staticGoal)
=======

        env.add_goal(splineGoal)
>>>>>>> ff805d4 (Fixes dynamic goal issues. Formats examples. Adds testing of examples. (#47))
    print("Starting episode")
    observation_history = []
    for i in range(n_steps):
        action, _states = model.predict(ob)
        ob, _, _, _ = env.step(action)
<<<<<<< HEAD
=======
        observation_history.append(ob)
        if i % 100 == 0:
            print(f"ob : {ob}")
    return observation_history
>>>>>>> ff805d4 (Fixes dynamic goal issues. Formats examples. Adds testing of examples. (#47))


if __name__ == "__main__":
    obstacles = True
    goal = True
    run_n_link_reacher(goal=goal, obstacles=obstacles)

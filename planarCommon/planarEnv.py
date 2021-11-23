from abc import abstractmethod
import numpy as np
from scipy.integrate import odeint

from gym import core
from gym.utils import seeding


class PlanarEnv(core.Env):

    metadata = {"render.modes": ["human", "rgb_array"], "video.frames_per_second": 15}

    def __init__(self, render=False, dt=0.01):
        self.viewer = None
        self.state = None
        self.setSpaces()
        self.seed()
        self._dt = dt
        self._render = render

    @abstractmethod
    def setSpaces(self):
        pass

    def dt(self):
        return self._dt

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    @abstractmethod
    def reset(self, pos=None, vel=None):
        pass

    @abstractmethod
    def step(self, a):
        pass

    @abstractmethod
    def _get_ob(self):
        pass

    @abstractmethod
    def _terminal(self):
        pass

    @abstractmethod
    def continuous_dynamics(self, x, t):
        pass

    def integrate(self):
        x0 = self.state[0 : 2 * self._n]
        t = np.arange(0, 2 * self._dt, self._dt)
        ynext = odeint(self.continuous_dynamics, x0, t)
        return ynext[1]

    @abstractmethod
    def render(self, mode="human"):
        pass

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None

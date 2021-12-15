import numpy as np
from numpy import sin, cos, pi
import time

from scipy.integrate import odeint

from gym import core, spaces
from gym.utils import seeding

from groundRobots.envs.groundRobotEnv import GroundRobotEnv


class GroundRobotVelEnv(GroundRobotEnv):
    def setSpaces(self):
        o = np.concatenate((self._limUpPos, self._limUpRelVel, self._limUpVel))
        self.observation_space = spaces.Dict({
            'x': spaces.Box(low=-self._limUpPos, high=self._limUpPos, dtype=np.float64),
            'xdot': spaces.Box(low=-self._limUpVel, high=self._limUpVel, dtype=np.float64),
            'vel': spaces.Box(low=-self._limUpRelVel, high=self._limUpRelVel, dtype=np.float64),
        })
        self.action_space = spaces.Box(
            low=-self._limUpRelVel, high=self._limUpRelVel, dtype=np.float64
        )

    def continuous_dynamics(self, x, t):
        # state = [x, y, theta, vel_for, vel_rel]
        x = self.state['x']
        xdot = np.array(
            [
                np.cos(x[2]) * self.action[0],
                np.sin(x[2]) * self.action[0],
                self.action[1],
            ]
        )
        self.state['xdot'] = xdot
        self.state['vel'] = self.action
        xddot = np.zeros(2)
        return np.concatenate((xdot, xddot))

import math
from gym.envs.classic_control import CartPoleEnv

class CartPoleEnvWrapper(CartPoleEnv):
    def __init__(self):
        super().__init__()
        self.theta_threshold_radians = 30 * 2 * math.pi / 360
        self.x_threshold = 5
        self.length = 1

    def step(self, action):
        state, reward, done, _, _ = super().step(action)
        if abs(state[2]) > self.theta_threshold_radians or abs(state[0]) > self.x_threshold:
            done = True
        reward = (30 - abs(math.degrees(state[2]))) / 30
        return state, reward, done, _, _

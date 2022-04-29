import numpy as np
from gym import spaces
from gym.core import Env


class BiClassEnvironment(Env):

    def __init__(self, data_x, data_y, pos_neg_ratio):
        super(BiClassEnvironment, self).__init__()
        if data_x.shape[0] != data_y.shape[0]:
            raise ValueError("len of data_x and data_y is not equal")
        self.data_x = data_x
        self.data_y = data_y
        self.reward_ratio = pos_neg_ratio

        self.num_classes = len(set(self.data_y))
        if self.num_classes != 2:
            raise ValueError("only 2 classes are allowed")

        self.episode_len = self.data_x.shape[0]
        self.index = np.arange(self.episode_len)
        self.action_space = spaces.Discrete(self.num_classes)
        self.time_step = 0

        self.actions = []

    def step(self, action):
        print("step function")

    def render(self, mode="human"):
        print("render function")

    def reset(self):
        print("reset function")



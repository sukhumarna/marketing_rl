from environment.env_mode import EnvMode
from gym import spaces
from gym.core import Env


class LeadConversionEnvironment(Env):

    def __init__(self, data_x, data_y, mode=EnvMode.TRAIN):
        super(LeadConversionEnvironment, self).__init__()
        if data_x.shape[0] != data_y.shape[0]:
            raise ValueError("len of data_x and data_y is not equal")
        self.data_x = data_x
        self.data_y = data_y
        self.mode = mode
        self.num_classes = len(set(self.data_y))
        if self.num_classes != 2:
            raise ValueError("only 2 classes are allowed")

    def reset(self):
        pass

    def render(self, mode="human"):
        pass

    def step(self, action):
        pass
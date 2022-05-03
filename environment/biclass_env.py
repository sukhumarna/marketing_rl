import numpy as np
from gym import spaces
from gym.core import Env
from sklearn.metrics import classification_report, roc_auc_score


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
        self.seed()

    def seed(self, seed=0):
        np.random.seed(seed)

    def step(self, action):
        terminal = False
        self.actions.append(action)
        # positive class
        if self.data_y[self.index[self.time_step]] == 1:
            # if correct prediction set reward to 1, otherwise -1
            if self.data_y[self.index[self.time_step]] == action:
                reward = 1
            else:
                reward = -1
                terminal = True
        # negative class
        else:
            # if correct prediction set reward to reward ratio, otherwise minus reward ratio
            if self.data_y[self.index[self.time_step]] == action:
                reward = self.reward_ratio
            else:
                reward = -self.reward_ratio

        self.time_step = self.time_step + 1

        # finish all samples, set terminal to true (finish the episode)
        next_state = None
        if self.time_step == self.episode_len:
            terminal = True
        else:
            next_state = self.data_x[self.index[self.time_step]]

        return next_state, reward, terminal

    def render(self, mode="human"):
        print('at time step {}/{}'.format(self.time_step, self.episode_len))
        print(classification_report(self.data_y[self.index], self.actions))
        rocauc = roc_auc_score(self.data_y[self.index], self.actions)
        print('rocauc score is', rocauc)
        return rocauc

    def reset(self):
        np.random.shuffle(self.index)
        self.time_step = 0
        self.actions = []
        return self.data_x[self.index[self.time_step]]



import numpy as np
from environment.env_mode import EnvMode
from gym import spaces
from gym.core import Env
from sklearn.metrics import classification_report


class BiClassEnvironment(Env):

    def __init__(self, data_x, data_y, pos_neg_ratio, mode=EnvMode.TRAIN):
        super(BiClassEnvironment, self).__init__()
        if data_x.shape[0] != data_y.shape[0]:
            raise ValueError("len of data_x and data_y is not equal")
        self.data_x = data_x
        self.data_y = data_y
        self.reward_ratio = pos_neg_ratio
        self.mode = mode

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
        info = {}
        terminal = False
        self.actions.append(action)
        # positive class
        if self.data_y[self.index[self.time_step]] == 1:
            # if correct prediction set reward to 1, otherwise -1
            if self.data_y[self.index[self.time_step]] == action:
                reward = 1
            else:
                reward = -1
                # TODO should stop the episode or not
                if self.mode == EnvMode.TRAIN:
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
        if self.time_step == self.episode_len:
            terminal = True
            report = self.info()
            label_1 = report['1']
            info['precision'], info['recall'] = label_1['precision'], label_1['recall']
            info['f1-score'] = label_1['f1-score']
            next_state = self.data_x[self.index[0]]
        else:
            next_state = self.data_x[self.index[self.time_step]]

        return next_state, reward, terminal, info

    def render(self, mode="human"):
        if self.time_step > 0 and self.time_step % 5000 == 0:
            print('at time step {}/{}'.format(self.time_step, self.episode_len))

    def reset(self):
        np.random.shuffle(self.index)
        self.time_step = 0
        self.actions = []
        return self.data_x[self.index[self.time_step]]

    def info(self):
        report = classification_report(self.data_y[self.index[:self.time_step]],
                                       self.actions,
                                       output_dict=True)
        return report

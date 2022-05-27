import numpy as np
from environment.env_mode import EnvMode
from gym import spaces
from gym.core import Env
from sklearn.metrics import classification_report


class FlexiBiClassEnvironment(Env):

    def __init__(self, data_x, data_y, pos_neg_ratio, reward, mode=EnvMode.TRAIN, early_stop=None, render_step=1000):
        super(FlexiBiClassEnvironment, self).__init__()
        if data_x.shape[0] != data_y.shape[0]:
            raise ValueError("len of data_x and data_y is not equal")
        self.data_x = data_x
        self.data_y = data_y
        self.reward_ratio = pos_neg_ratio
        self.reward = reward
        self.mode = mode
        self.pos_miss_count = 0
        self.num_classes = len(set(self.data_y))
        if self.num_classes != 2:
            raise ValueError("only 2 classes are allowed")
        self.episode_len = self.data_x.shape[0]
        self.index = np.arange(self.episode_len)
        self.action_space = spaces.Discrete(self.num_classes)
        self.time_step = 0
        self.early_stop = early_stop
        self.render_step = render_step

        self.actions = []
        self.seed()

    def _get_index(self):
        if self.time_step >= len(self.index):
            return self.index[0]
        return self.index[self.time_step]

    def _positive(self) -> bool:
        return self.data_y[self._get_index()] == 1

    def _correct_prediction(self, action) -> bool:
        return self.data_y[self._get_index()] == action

    def seed(self, seed=0):
        np.random.seed(seed)

    def step(self, action):
        info = {}
        terminal = False
        self.actions.append(action)
        # positive class
        if self._positive():
            if self._correct_prediction(action):
                reward = self.reward[self._get_index()]
            else:
                reward = -1
                if self.mode == EnvMode.TRAIN and self.early_stop is not None:
                    self.pos_miss_count += 1
                    if self.pos_miss_count > self.early_stop:
                        terminal = True
        # negative class
        else:
            if self._correct_prediction(action):
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
            next_state = self.data_x[self._get_index()]

        return next_state, reward, terminal, info

    def reset(self):
        np.random.shuffle(self.index)
        self.time_step = 0
        self.actions = []
        self.pos_miss_count = 0
        return self.data_x[self._get_index()]

    def render(self, mode="human"):
        if self.time_step > 0 and self.time_step % self.render_step == 0:
            print('at time step {}/{}'.format(self.time_step, self.episode_len))

    def info(self):
        report = classification_report(self.data_y[self.index[:self.time_step]],
                                       self.actions,
                                       output_dict=True)
        return report

import numpy as np
from typing import Any
from tf_agents.environments import py_environment
from tf_agents.trajectories import time_step as ts
from tf_agents.typing import types
from tf_agents.specs import array_spec
from environment.env_mode import EnvMode


class BiClassTFEnv(py_environment.PyEnvironment):
    def __init__(self, data_x, data_y, pos_neg_ratio, mode=EnvMode.TRAIN, discount=0.5, early_stop=None, seed=0):
        super(BiClassTFEnv, self).__init__()
        num_classes = len(set(data_y))
        if num_classes != 2:
            raise ValueError("only 2 classes are allowed")
        input_size = data_x.shape[1]
        self._data_x = data_x
        self._data_y = data_y
        self._reward_ratio = pos_neg_ratio
        self._mode = mode
        self._discount = discount
        self._early_stop = early_stop
        self._action_spec = array_spec.BoundedArraySpec(shape=(), dtype=np.int32, minimum=0, maximum=1, name='action')
        self._observation_spec = array_spec.ArraySpec(shape=(input_size,), dtype=float, name='observation')
        self._episode_len = self._data_x.shape[0]
        self._index = np.arange(self._episode_len)
        self._time_step = 0
        self._state = data_x[self._get_index()]
        self._episode_ended = False
        self._pos_miss_count = 0
        self.seed(seed=seed)

    def seed(self, seed=0):
        np.random.seed(seed)

    def _get_index(self):
        if self._time_step >= len(self._index):
            return self._index[0]
        return self._index[self._time_step]

    def observation_spec(self) -> types.NestedArraySpec:
        return self._observation_spec

    def action_spec(self) -> types.NestedArraySpec:
        return self._action_spec

    # def get_info(self) -> types.NestedArray:
    #     pass
    #
    # def get_state(self) -> Any:
    #     pass
    #
    # def set_state(self, state: Any) -> None:
    #     pass

    def _step(self, action: types.NestedArray) -> ts.TimeStep:
        if self._episode_ended:
            return self.reset()

        true_label = self._data_y[self._get_index()]
        # for positive label
        if true_label == 1:
            if true_label == action:
                reward = 1
            else:
                reward = -1
                if self._mode == EnvMode.TRAIN and self._early_stop is not None:
                    self._pos_miss_count += 1
                    if self._pos_miss_count >= self._early_stop:
                        self._episode_ended = True
        # for negative label
        else:
            if true_label == action:
                reward = self._reward_ratio
            else:
                reward = -self._reward_ratio
        self._time_step += 1
        if self._time_step == self._episode_len:
            self._episode_ended = True

        if self._episode_ended:
            return ts.termination(np.array(self._data_x[self._get_index()], dtype=float), reward=reward)
        else:
            return ts.transition(np.array(self._data_x[self._get_index()], dtype=float),
                                 reward=reward, discount=self._discount)

    def _reset(self) -> ts.TimeStep:
        np.random.shuffle(self._index)
        self._time_step = 0
        self._pos_miss_count = 0
        self._episode_ended = False
        self._state = self._data_x[self._get_index()]
        return ts.restart(np.array(self._state, dtype=float))

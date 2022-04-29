import numpy as np
import pandas as pd
import unittest
from gym import spaces
from environment.biclass_env import BiClassEnvironment


class TestBiClassEnvironment(unittest.TestCase):

    def test_biclass_environment(self):
        data = {'data1': [0.5, 0.7, 0.3, 0.6, 0.1], 'data2': [0.32, 0.56, 0.27, 0.74, 0.25]}
        df = pd.DataFrame(data, columns=['data1', 'data2'])
        data_x = df.to_numpy()
        data_y = np.array([1, 0, 0, 0, 1])
        pos_neg_ratio = 2.0/3.0

        env = BiClassEnvironment(data_x, data_y, pos_neg_ratio)
        self.assertEqual(env.episode_len, 5)
        self.assertEqual(env.num_classes, 2)
        self.assertEqual(env.action_space, spaces.Discrete(2))

    def test_biclass_environment_error(self):
        data = {'data1': [0.5, 0.7, 0.3, 0.6, 0.1], 'data2': [0.32, 0.56, 0.27, 0.74, 0.25]}
        df = pd.DataFrame(data, columns=['data1', 'data2'])
        data_x = df.to_numpy()
        data_y = np.array([1, 0, 0, 0])
        pos_neg_ratio = 2.0/3.0
        self.assertRaises(ValueError, BiClassEnvironment,
                          data_x=data_x, data_y=data_y, pos_neg_ratio=pos_neg_ratio)
        data_y = np.array([1, 0, 0, 0, 2])
        self.assertRaises(ValueError, BiClassEnvironment,
                          data_x=data_x, data_y=data_y, pos_neg_ratio=pos_neg_ratio)


if __name__ == '__main__':
    unittest.main()


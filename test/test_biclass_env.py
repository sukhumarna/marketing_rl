import numpy as np
import pandas as pd
import unittest
from gym import spaces
from environment.biclass_env import BiClassEnvironment, EnvMode


class TestBiClassEnvironment(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.data_x = np.array([[0.1, 1.0], [0.2, 2.0], [0.3, 3.0], [0.4, 4.0], [0.5, 5.0]])
        cls.data_y = np.array([1, 0, 0, 1, 0])
        cls.pos_neg_ratio = 2.0/3.0

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

    def test_step(self):
        env = BiClassEnvironment(self.data_x, self.data_y, self.pos_neg_ratio)
        # set index in order to test in sequence
        env.index = np.arange(5)

        # case correct positive class
        next_state, reward, terminal, info = env.step(action=1)
        self.assertEqual(reward, 1)
        self.assertTrue(np.array_equal(np.array([0.2, 2.0]), next_state))
        self.assertFalse(terminal)

        # case correct negative class
        next_state, reward, terminal, info = env.step(action=0)
        self.assertAlmostEqual(reward, self.pos_neg_ratio, places=2)
        self.assertFalse(terminal)

        # case incorrect negative class (action=1, actual=0)
        next_state, reward, terminal, info = env.step(action=1)
        self.assertAlmostEqual(reward, -self.pos_neg_ratio, places=2)
        self.assertFalse(terminal)

        # case incorrect positive class (action=0, actual=1)
        next_state, reward, terminal, info = env.step(action=0)
        self.assertEqual(reward, -1)
        self.assertTrue(terminal)

        # correct prediction, last step in the episode
        next_state, reward, terminal, info = env.step(action=0)
        self.assertAlmostEqual(reward, self.pos_neg_ratio, places=2)
        self.assertTrue(terminal)
        self.assertTrue(np.array_equal(np.array([0.1, 1.0]), next_state))

        env = BiClassEnvironment(self.data_x, self.data_y, self.pos_neg_ratio, mode=EnvMode.TEST)
        next_state, reward, terminal, info = env.step(action=0)
        self.assertEqual(reward, -1)
        self.assertFalse(terminal)

    def test_reset(self):
        env = BiClassEnvironment(self.data_x, self.data_y, self.pos_neg_ratio)

        env.seed(seed=1)
        env.step(action=1)
        env.step(action=1)
        self.assertTrue(np.array_equal(np.array([1, 1]), env.actions))
        self.assertEqual(env.time_step, 2)

        state = env.reset()
        self.assertTrue(np.array_equal(np.array([]), env.actions))
        self.assertEqual(env.time_step, 0)
        self.assertTrue(np.array_equal(self.data_x[env.index[0]], state))

    def test_render(self):
        env = BiClassEnvironment(self.data_x, self.data_y, self.pos_neg_ratio)
        env.index = np.arange(5)
        # actual:        [1, 0, 0, 1, 0]
        # prediction:    [1, 0, 0, 0, 1]
        actions = [1, 0, 0, 0, 1]
        for a in actions:
            env.step(action=a)
        env.render()

        env = BiClassEnvironment(self.data_x, self.data_y, self.pos_neg_ratio)
        env.index = [1, 2, 3, 0, 4]
        actions = [0, 0, 0, 1, 1]
        for a in actions:
            env.step(action=a)

    def test_info(self):
        env = BiClassEnvironment(self.data_x, self.data_y, self.pos_neg_ratio)
        env.index = np.arange(5)
        # actual:        [1, 0, 0, 1, 0]
        # prediction:    [1, 0, 0, 0, 1]
        actions = [1, 0, 0, 0, 1]
        for a in actions:
            env.step(action=a)
        report = env.info()
        label_1 = report['1']
        self.assertAlmostEqual(label_1['recall'], 0.50, places=2)
        self.assertAlmostEqual(label_1['precision'], 0.50, places=2)


if __name__ == '__main__':
    unittest.main()


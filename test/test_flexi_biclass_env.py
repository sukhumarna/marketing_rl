import numpy as np
import unittest
from environment.flexi_biclass_env import FlexiBiClassEnvironment


class TestFlexiBiClassEnvironment(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.data_x = np.array([[0.1, 1.0], [0.2, 2.0], [0.3, 3.0], [0.4, 4.0], [0.5, 5.0]])
        cls.data_y = np.array([1, 0, 0, 1, 0])
        cls.reward = np.array([50, 0, 0, 100, 0])
        cls.pos_neg_ratio = 2.0/3.0

    def test_get_index(self):
        env = FlexiBiClassEnvironment(data_x=self.data_x, data_y=self.data_y, pos_neg_ratio=self.pos_neg_ratio,
                                      reward=self.reward, early_stop=None)
        env.index = [4, 2, 1, 0, 3]
        self.assertEqual(env._get_index(), 4)
        env.step(action=0)
        self.assertEqual(env._get_index(), 2)
        env.step(action=0)
        self.assertEqual(env._get_index(), 1)
        env.step(action=0)
        self.assertEqual(env._get_index(), 0)
        env.step(action=1)
        self.assertEqual(env._get_index(), 3)
        env.step(action=1)
        self.assertEqual(env._get_index(), 4)

    def test_positive(self):
        env = FlexiBiClassEnvironment(data_x=self.data_x, data_y=self.data_y, pos_neg_ratio=self.pos_neg_ratio,
                                      reward=self.reward, early_stop=None)

        # label: 1, 0, 0, 1, 0
        self.assertTrue(env._positive())
        env.step(action=0)
        self.assertFalse(env._positive())
        env.step(action=0)
        self.assertFalse(env._positive())
        env.step(action=0)
        self.assertTrue(env._positive())
        env.step(action=1)
        self.assertFalse(env._positive())
        env.step(action=0)
        self.assertTrue(env._positive())

    def test_correct_prediction(self):
        env = FlexiBiClassEnvironment(data_x=self.data_x, data_y=self.data_y, pos_neg_ratio=self.pos_neg_ratio,
                                      reward=self.reward, early_stop=None)
        # label: 1, 0, 0, 1, 0
        self.assertTrue(env._correct_prediction(action=1))
        env.step(action=1)
        self.assertFalse(env._correct_prediction(action=1))
        env.step(action=0)
        self.assertTrue(env._correct_prediction(action=0))
        env.step(action=0)
        self.assertFalse(env._correct_prediction(action=0))
        env.step(action=1)
        self.assertTrue(env._correct_prediction(action=0))
        env.step(action=0)
        self.assertFalse(env._correct_prediction(action=0))

    def test_step(self):
        env = FlexiBiClassEnvironment(data_x=self.data_x, data_y=self.data_y, pos_neg_ratio=self.pos_neg_ratio,
                                      reward=self.reward, early_stop=0)
        # label: 1, 0, 0, 1, 0
        # reward: 50, 0, 0, 100, 0

        # case correct positive class
        next_state, reward, terminal, info = env.step(action=1)
        self.assertEqual(reward, 50)
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

        env = FlexiBiClassEnvironment(data_x=self.data_x, data_y=self.data_y, pos_neg_ratio=self.pos_neg_ratio,
                                      reward=self.reward, early_stop=None)
        for i in range(4):
            _, _, terminal, _ = env.step(action=0)
            self.assertFalse(terminal)

    def test_reset(self):
        env = FlexiBiClassEnvironment(data_x=self.data_x, data_y=self.data_y, pos_neg_ratio=self.pos_neg_ratio,
                                      reward=self.reward, early_stop=5)

        env.step(action=0)
        env.step(action=0)
        self.assertEqual(env._get_index(), 2)
        self.assertEqual(env.pos_miss_count, 1)
        self.assertEqual(env.actions, [0, 0])
        self.assertEqual(env.time_step, 2)

        state = env.reset()
        self.assertEqual(env.actions, [])
        self.assertEqual(env.time_step, 0)
        self.assertTrue(np.array_equal(self.data_x[env._get_index()], state))

    def test_render(self):
        env = FlexiBiClassEnvironment(data_x=self.data_x, data_y=self.data_y, pos_neg_ratio=self.pos_neg_ratio,
                                      reward=self.reward, early_stop=None, render_step=2)
        env.step(action=0)
        env.step(action=0)
        env.render(mode='human')


if __name__ == '__main__':
    unittest.main()

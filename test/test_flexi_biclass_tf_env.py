import numpy as np
import unittest
from tf_agents.environments import utils
from environment.flexi_biclass_tf_env import FlexiBiClassTFEnv


class FlexiBiClassTFEnvTestCase(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.data_x = np.array([[0.1, 1.0], [0.2, 2.0], [0.3, 3.0], [0.4, 4.0], [0.5, 5.0]])
        cls.data_y = np.array([1, 0, 0, 1, 0])
        cls.reward = np.array([1.5, 0, 0, 2, 0])
        cls.pos_neg_ratio = 2.0/3.0

    def test_flexi_biclass_tf_en(self):
        env = FlexiBiClassTFEnv(data_x=self.data_x, data_y=self.data_y,
                                reward=self.reward, pos_neg_ratio=self.pos_neg_ratio)
        utils.validate_py_environment(env, episodes=5)


if __name__ == '__main__':
    unittest.main()

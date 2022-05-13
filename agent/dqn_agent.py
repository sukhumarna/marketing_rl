import numpy as np
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam


class DQNAgent:

    def __init__(self, input_size, action_size, learning_rate, gamma, verbose=0):
        self.input_size = input_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.verbose = verbose

        self.qnet = self.build_qnetwork(self.input_size, self.learning_rate)
        self.target_qnet = self.build_qnetwork(self.input_size, self.learning_rate)

    @staticmethod
    def build_qnetwork(input_size, learning_rate):
        model = Sequential()
        model.add(Dense(input_dim=input_size, activation='relu'))
        model.add(Dense(16, activation='relu'))
        model.add(Dense(8, activation='relu'))
        model.add(Dense(2, activation='linear'))
        model.compile(optimizer=Adam(learning_rate=learning_rate), loss='mse')
        return model

    def get_action(self, state):
        state_tensor = tf.convert_to_tensor(state[None, :], dtype=tf.float32)
        action_q = self.qnet(state_tensor)
        action = np.argmax(action_q.numpy()[0], axis=0)
        if self.verbose == 1:
            print("state: ", state)
            print("q values:", action_q)
            print("optimal action:", action)
        return action

    def train(self, batch):
        states, actions, rewards, next_states, dones = batch
        current_q = self.qnet(states)
        next_q = self.target_qnet(next_states)
        pass

class ReplayBuffer:
    def __init__(self, max_size):
        self.max_size = max_size

    def store_experience(self, state, action, reward, next_state, done):
        pass

    def sample_experiences(self, batch_size):
        pass
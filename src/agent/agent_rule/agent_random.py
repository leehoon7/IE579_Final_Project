import numpy as np


class AgentRandom():
    def __init__(self, num_agent, dim_obs, dim_action):
        self.num_agent = 5
        self.dim_action = dim_action

    def get_action(self, obs):
        return np.random.randint(self.dim_action, size=(self.num_agent), dtype=np.int32)

    def get_action_eval(self, obs):
        return np.random.randint(self.dim_action, size=(1), dtype=np.int32)

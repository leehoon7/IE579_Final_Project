import numpy as np
import matplotlib.pyplot as plt

import magent
from visualizer import CustomVisualizer


class MAgentBattle():

    def __init__(self, visualize, eval_mode):

        self.visualize = visualize
        self.eval_mode = eval_mode

        map_size = 15
        self.env = magent.GridWorld("battle_small", map_size=map_size + 2)
        self.handles = self.env.get_handles()  # ID of each team.
        self.num_team = 2
        self.num_agent = 5
        self.env_t = 0
        self.env_t_max = 300
        self.self_pos = self.env.view_space[0][0]//2
        self.action_dim = self.env.get_action_space(self.handles[1])[0]

        if self.visualize:
            self.env.set_render_dir("../build/render")
            self.visualizer = CustomVisualizer(map_size=map_size)

    def reset(self):

        self.env.reset()
        self.env.add_agents(self.handles[0], method="random", n=self.num_agent)  # add team 1 (RED)
        self.env.add_agents(self.handles[1], method="random", n=self.num_agent)  # add team 2 (BLUE)
        self.env_t = 0

        obs1 = [
            self.env.get_observation(self.handles[0]),
            self.env.get_agent_id(self.handles[0]),
        ]

        obs2 = [
            self.env.get_observation(self.handles[1]),
            self.env.get_agent_id(self.handles[1])
        ]

        if self.visualize:
            self.visualizer.reset(self.env.get_pos(self.handles[0]), self.env.get_pos(self.handles[1]),
                                  ids=[obs1[1], obs2[1]])

        return (obs1, obs2), False  # observation, termination info

    def step(self, action1, action2):

        self.env_t += 1

        # 1. apply action
        self.env.set_action(self.handles[0], action1)
        self.env.set_action(self.handles[1], action2)
        done = self.env.step()
        self.env.clear_dead()

        # 2. render info
        self.env.render()

        # 3. get observation
        obs1 = [
            self.env.get_observation(self.handles[0]),
            self.env.get_agent_id(self.handles[0]),
        ]

        obs2 = [
            self.env.get_observation(self.handles[1]),
            self.env.get_agent_id(self.handles[1])
        ]

        # 4. get reward
        rewards = [
            self.env.get_reward(self.handles[0]),
            self.env.get_reward(self.handles[1])
        ]

        if self.env_t > self.env_t_max:
            done = True

        if self.visualize:
            hp1 = obs1[0][0][:, self.self_pos, self.self_pos, 2]
            hp2 = obs2[0][0][:, self.self_pos, self.self_pos, 2]
            self.visualizer.step(self.env.get_pos(self.handles[0]), self.env.get_pos(self.handles[1]),
                                 hp1=hp1, hp2=hp2, act1=action1, act2=action2, ids=[obs1[1], obs2[1]])

        return (obs1, obs2), rewards, done

    def close(self):
        if self.visualize:
            self.visualizer.close()

    def _get_random_action(self):
        return np.random.randint(self.action_dim, size=(self.num_agent), dtype=np.int32)


if __name__ == "__main__":

    env = MAgentBattle(visualize=True, eval_mode=False)
    obs, done = env.reset()

    while not done:
        a1 = env._get_random_action()
        a2 = env._get_random_action()
        (obs1, obs2), reward, done = env.step(a1, a2)

    env.close()

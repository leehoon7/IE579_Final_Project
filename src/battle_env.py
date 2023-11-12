import numpy as np
import matplotlib.pyplot as plt

import magent
from visualizer import CustomVisualizer


class MAgentBattle():

    def __init__(self, visualize, eval_mode, obs_flat=False):

        self.visualize = visualize
        self.eval_mode = eval_mode
        self.obs_flat = obs_flat

        map_size = 15
        self.env = magent.GridWorld("battle_small", map_size=map_size + 2)
        self.handles = self.env.get_handles()  # ID of each team.
        self.num_team = 2
        self.num_agent = 5
        self.env_t = 0
        self.env_t_max = 300
        self.self_pos = self.env.view_space[0][0]//2
        self.dim_obs = 11 * 11 * 7 + 34  # 11 X 11 with 7 channels + 34 additional features
        self.dim_action = self.env.get_action_space(self.handles[1])[0]

        self.done_before = (None, None)

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

        if self.obs_flat and not self.eval_mode:
            alive_agent1 = len(obs1[1])
            obs1[0] = np.concatenate([obs1[0][0].reshape(alive_agent1, -1), obs1[0][1]], axis=-1)

            alive_agent2 = len(obs2[1])
            obs2[0] = np.concatenate([obs2[0][0].reshape(alive_agent2, -1), obs2[0][1]], axis=-1)

        if self.eval_mode:
            obs1 = {
                agent_id: (obs1[0][0][i], obs1[0][1][i]) for i, agent_id in enumerate(obs1[1])
            }
            obs2 = {
                agent_id - self.num_agent: (obs2[0][0][i], obs2[0][1][i]) for i, agent_id in enumerate(obs2[1])
            }

        done1, done2 = np.zeros(self.num_agent), np.zeros(self.num_agent)
        self.done_before = (done1, done2)
        valid1, valid2 = np.ones(self.num_agent), np.ones(self.num_agent)

        return (obs1, obs2), (done1, done2, False), (valid1, valid2)  # observation, termination info

    def step(self, action1, action2):

        self.env_t += 1

        # 1. apply action
        self.env.set_action(self.handles[0], action1)
        self.env.set_action(self.handles[1], action2)
        env_done = self.env.step()

        # 2. get reward
        reward1, reward2 = self.env.get_reward(self.handles[0]), self.env.get_reward(self.handles[1])

        # 3. clear dead agent & render (official rendering)
        self.env.clear_dead()
        if self.visualize:
            self.env.render()

        # 4. get observation
        obs1_ = [np.zeros((5, 11, 11, 7), dtype=np.float32), np.zeros((5, 34), dtype=np.float32)]
        obs1_id = self.env.get_agent_id(self.handles[0])
        obs1_orig = self.env.get_observation(self.handles[0])
        obs1_[0][obs1_id] = obs1_orig[0]
        obs1_[1][obs1_id] = obs1_orig[1]
        obs1 = [obs1_, obs1_id]

        obs2_ = [np.zeros((5, 11, 11, 7), dtype=np.float32), np.zeros((5, 34), dtype=np.float32)]
        obs2_id = self.env.get_agent_id(self.handles[1])
        obs2_orig = self.env.get_observation(self.handles[1])
        obs2_[0][obs2_id - self.num_agent] = obs2_orig[0]
        obs2_[1][obs2_id - self.num_agent] = obs2_orig[1]
        obs2 = [obs2_, obs2_id]

        if self.visualize:
            hp1 = obs1[0][0][:, self.self_pos, self.self_pos, 2]
            hp2 = obs2[0][0][:, self.self_pos, self.self_pos, 2]
            self.visualizer.step(self.env.get_pos(self.handles[0]), self.env.get_pos(self.handles[1]),
                                 hp1=hp1, hp2=hp2, act1=action1, act2=action2, ids=[obs1[1], obs2[1]])

        if self.obs_flat and not self.eval_mode:
            obs1[0] = np.concatenate([obs1[0][0].reshape(self.num_agent, -1), obs1[0][1]], axis=-1)
            obs2[0] = np.concatenate([obs2[0][0].reshape(self.num_agent, -1), obs2[0][1]], axis=-1)

        done1 = np.ones(self.num_agent)
        done1[obs1[1]] = False
        done2 = np.ones(self.num_agent)
        done2[obs2[1] - self.num_agent] = False

        done_before1, done_before2 = self.done_before
        valid1, valid2 = 1 - done1 * done_before1, 1 - done2 * done_before2
        self.done_before = (done1, done2)

        if self.env_t > self.env_t_max:
            env_done = True
            done1 = np.ones(self.num_agent)
            done2 = np.ones(self.num_agent)

        if self.eval_mode:
            obs1 = {
                agent_id: (obs1[0][0][i], obs1[0][1][i]) for i, agent_id in enumerate(obs1[1])
            }
            obs2 = {
                agent_id - self.num_agent: (obs2[0][0][i], obs2[0][1][i]) for i, agent_id in enumerate(obs2[1])
            }

        return (obs1, obs2), (reward1, reward2), (done1, done2, env_done), (valid1, valid2)

    def close(self):
        if self.visualize:
            self.visualizer.close()

    def _get_random_action(self):
        return np.random.randint(self.dim_action, size=(self.num_agent), dtype=np.int32)


if __name__ == "__main__":

    env = MAgentBattle(visualize=False, eval_mode=False)
    (obs1, obs2), (done1, done2, done_env), (valid1, valid2) = env.reset()

    while not done_env:
        a1 = env._get_random_action()
        a2 = env._get_random_action()
        (obs1, obs2), (reward1, reward2), (done1, done2, done_env), (valid1, valid2) = env.step(a1, a2)

    env.close()

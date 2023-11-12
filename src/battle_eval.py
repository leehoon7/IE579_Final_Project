import time

import numpy as np

from battle_env import MAgentBattle
from agent.agent_rl.agent_rl import AgentRL
from agent.agent_rule.agent_random import AgentRandom


if __name__ == "__main__":

    ########## You have to design 'get_action_eval' for evaluation.
    ########## This script is evaluation for 1 episode.

    env = MAgentBattle(visualize=False, eval_mode=True)

    agent1 = [AgentRandom(num_agent=1, dim_obs=env.dim_obs, dim_action=env.dim_action) for _ in range(env.num_agent)]
    agent2 = [AgentRandom(num_agent=1, dim_obs=env.dim_obs, dim_action=env.dim_action) for _ in range(env.num_agent)]

    (obs1, obs2), (done1, done2, done_env), (valid1, valid2) = env.reset()

    start = time.time()
    env_t = 0
    while not done_env:

        # Team 1 make decisions. (in a decentralized manner)
        a1 = []
        for agent_id, obs in obs1.items():
            a1.append(agent1[agent_id].get_action_eval(obs))
        a1 = np.concatenate(a1)

        # Team 2 make decisions. (in a decentralized manner)
        a2 = []
        for agent_id, obs in obs2.items():
            a2.append(agent2[agent_id].get_action_eval(obs))
        a2 = np.concatenate(a2)

        (obs1, obs2), reward, (done1, done2, done_env), (valid1, valid2) = env.step(a1, a2)
        env_t += 1

    env.close()

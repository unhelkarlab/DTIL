import os
import numpy as np
import functools
from gymnasium.spaces import Box, Discrete
from .haven_envs.starcraft2.starcraft2 import StarCraft2Env
from pettingzoo.utils.env import ParallelEnv
from smac.env.starcraft2.maps import get_map_params


class SMAC_V1(ParallelEnv):
  metadata = {"render_modes": ["human"], "name": "smac_v1"}

  def __init__(self, map_name) -> None:
    super().__init__()
    self.env = StarCraft2Env(map_name=map_name)

    self.max_length = get_map_params(map_name)["limit"]
    self.possible_agents = list(range(self.env.n_agents))
    self.agents = self.possible_agents[:]

    observation_size = self.env.get_obs_size()
    self.observation_spaces = {}
    for name in self.agents:
      self.observation_spaces[name] = Box(low=-1,
                                          high=1,
                                          shape=(observation_size, ),
                                          dtype=np.float32)

    n_actions = self.env.get_total_actions()
    self.action_spaces = {}
    for name in self.agents:
      self.action_spaces[name] = Discrete(n_actions)

  @functools.lru_cache(maxsize=None)
  def observation_space(self, agent):
    return self.observation_spaces[agent]

  @functools.lru_cache(maxsize=None)
  def action_space(self, agent):
    return self.action_spaces[agent]

  def close(self):
    self.env.close()

  def reset(self, seed=None, options=None):
    if seed is not None:
      self.env._seed = seed

    obs, state = self.env.reset()

    self.agents = self.possible_agents[:]
    self.cur_step = 0

    infos = []
    avail_actions = self.env.get_avail_actions()
    for idx in range(self.env.n_agents):
      infos.append({"avail_actions": np.array(avail_actions[idx])})

    return obs, infos

  def step(self, actions):
    list_actions = []
    for aname in self.agents:
      list_actions.append(actions[aname])

    reward, terminated, info = self.env.step(list_actions)

    obs = self.env.get_obs()
    avail_actions = self.env.get_avail_actions()

    new_info = {}
    for aname in self.agents:
      new_info[aname] = {}
      # copy same info to all agents
      for key, val in info.items():
        if key == "battle_won":
          key = "won"
        new_info[aname][key] = val

      new_info[aname]["avail_actions"] = np.array(avail_actions[aname])

    self.cur_step += 1

    trunc = self.cur_step >= self.max_length
    if terminated:
      dones = {agent: True for agent in self.agents}
      truncs = {agent: False for agent in self.agents}
    else:
      dones = {agent: False for agent in self.agents}
      truncs = {agent: trunc for agent in self.agents}

    rewards = {agent: reward for agent in self.agents}

    return obs, rewards, dones, truncs, new_info

  def save_replay(self):
    self.env.save_replay()


if __name__ == "__main__":
  cur_dir = os.path.dirname(__file__)

  env = SMAC_V1("2s3z")
  env_eval = SMAC_V1("2s3z")

  epi_step = 0
  obs, infos = env.reset(seed=0)
  done = False
  while not done:
    actions = {}
    for agent in env.agents:
      avail_actions = infos[agent]["avail_actions"]
      avail_actions_ind = np.nonzero(avail_actions)[0]
      action = np.random.choice(avail_actions_ind)
      actions[agent] = action

    obs, rewards, dones, truncs, infos = env.step(actions)
    done = all(dones.values())
    epi_step += 1
  env.close()
  print(epi_step)

import os
from typing import Type
import pickle
import functools
from collections import defaultdict
import gymnasium
from gymnasium.spaces import Discrete, Box
from gymnasium.utils import seeding
import numpy as np
import cv2
from PIL import Image
from pettingzoo import ParallelEnv
from .labor_division import LaborDivision, LDExpert_V2


class LaborDivisionV2(LaborDivision):

  @functools.lru_cache(maxsize=None)
  def observation_space(self, agent):
    # gymnasium spaces are defined and documented here:
    #    https://gymnasium.farama.org/api/spaces/
    n_agents = len(self.possible_agents)
    obs_low = np.array([*self.world_low, 0] +
                       [0, -self.vis_rad, -self.vis_rad] * (n_agents - 1))
    obs_high = np.array([*self.world_high, 1] +
                        [1, self.vis_rad, self.vis_rad] * (n_agents - 1))

    return Box(low=obs_low, high=obs_high, dtype=np.float32)

  def _compute_obs(self, actions):
    dict_obs = {}
    for agent_me in self.possible_agents:
      my_pos = self.dict_agent_pos[agent_me]
      obs = [my_pos]

      # progressing?
      progressing, _ = self._get_progressing_state(agent_me)
      obs.append(progressing)

      # relative pos of other agents
      for agent_fr in self.possible_agents:
        if agent_me == agent_fr:
          continue
        # relative position. if out of visible range, set it as not-observed
        rel_pos = self.dict_agent_pos[agent_fr] - my_pos
        fr_state = [1, *rel_pos]
        if np.linalg.norm(rel_pos) > self.vis_rad:
          fr_state = [0, 0, 0]

        obs.append(fr_state)

      dict_obs[agent_me] = np.hstack(obs)

    return dict_obs


class DyadLaborDivisionV2(LaborDivisionV2):

  def __init__(self, targets, render_mode=None):
    super().__init__(targets, n_agents=2, render_mode=render_mode)


class TwoTargetDyadLaborDivisionV2(DyadLaborDivisionV2):

  def __init__(self, render_mode=None):
    super().__init__([(-4, 0), (4, 0)], render_mode)


class ThreeTargetDyadLaborDivisionV2(DyadLaborDivisionV2):

  def __init__(self, render_mode=None):
    super().__init__([(-4, -3), (4, -3), (0, 4)], render_mode)


class LDv2Expert(LDExpert_V2):

  def conv_obs(self, obs):
    pos = obs[0:2]
    progressing = obs[2]

    # let's think about only dyad setting
    observed = obs[3]
    rel_pos = obs[4:6]
    return pos, None, progressing, observed, rel_pos, None


def generate_data(save_dir,
                  expert_class: Type[LDv2Expert],
                  env_name,
                  n_traj,
                  render=False,
                  render_delay=10):
  expert_trajs = defaultdict(list)
  if env_name == "LaborDivision2-v2":
    env = TwoTargetDyadLaborDivisionV2(render_mode="human")
  elif env_name == "LaborDivision3-v2":
    env = ThreeTargetDyadLaborDivisionV2(render_mode="human")
  else:
    raise NotImplementedError()

  env.set_render_delay(render_delay)
  agents = {
      aname: expert_class(env, env.tolerance, aname)
      for aname in env.possible_agents
  }

  list_total_reward = []
  for _ in range(n_traj):
    obs, infos = env.reset()
    prev_latents = {aname: agents[aname].PREV_LATENT for aname in env.agents}
    episode_reward = {aname: 0 for aname in env.agents}

    total_reward = 0
    samples = []
    while True:
      actions = {}
      latents = {}
      for aname in env.agents:
        latent = agents[aname].choose_mental_state(obs[aname],
                                                   prev_latents[aname])
        action = agents[aname].choose_policy_action(obs[aname], latent)
        actions[aname] = action
        latents[aname] = latent

      next_obs, rewards, dones, truncs, infos = env.step(actions)
      samples.append((obs, actions, next_obs, latents, rewards, dones))

      for aname in env.agents:
        episode_reward[aname] += rewards[aname]
        total_reward += rewards[aname]

      prev_latents = latents
      obs = next_obs

      if render:
        env.render()

      if all(truncs.values()) or all(dones.values()):
        break

    list_total_reward.append(total_reward)
    if render:
      print(episode_reward, total_reward, len(samples))

    # structure: (traj_length, num_agents, value_dim)
    all_obs, all_act, all_n_obs, all_lat, all_rew, all_don = list(zip(*samples))

    # transpose structure: (num_agents, traj_length, value_dim)
    def transpose_impl(all_item):
      return [[item[aname] for item in all_item] for aname in env.agents]

    expert_trajs["states"].append(transpose_impl(all_obs))
    expert_trajs["next_states"].append(transpose_impl(all_n_obs))
    expert_trajs["actions"].append(transpose_impl(all_act))
    expert_trajs["latents"].append(transpose_impl(all_lat))
    expert_trajs["rewards"].append(transpose_impl(all_rew))
    expert_trajs["dones"].append(transpose_impl(all_don))
    expert_trajs["lengths"].append(len(all_obs))

  print("Mean:", np.mean(list_total_reward))
  if save_dir is not None:
    file_path = os.path.join(save_dir, f"{env_name}_{n_traj}.pkl")
    with open(file_path, 'wb') as f:
      pickle.dump(expert_trajs, f)

  return expert_trajs


if __name__ == "__main__":
  cur_dir = os.path.dirname(__file__)

  # traj = generate_data(cur_dir, LDv2Expert, "LaborDivision2-v2", 100, False, 10)
  # traj = generate_data(cur_dir, LDv2Expert, "LaborDivision3-v2", 50, False, 10)
  traj = generate_data(None, LDv2Expert, "LaborDivision2-v2", 1, True, 5000)
  # traj = generate_data(None, LDv2Expert, "LaborDivision3-v2", 50, True, 1000)

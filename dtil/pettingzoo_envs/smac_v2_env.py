import os
import pickle
import numpy as np
import random
import torch
from onpolicy.envs.starcraft2.SMACv2_modified import SMACv2
from pettingzoo.utils.env import ParallelEnv  # noqa: F401
import functools
from omegaconf import OmegaConf, DictConfig
from onpolicy.algorithms.r_mappo.algorithm.r_actor_critic import R_Actor
from collections import defaultdict
from gymnasium.spaces import Box, Discrete
# from gym.spaces import Box


def parse_smacv2_distribution(n_allies, n_enemies, map_name):
  distribution_config = {
      "n_units": int(n_allies),
      "n_enemies": int(n_enemies),
      "start_positions": {
          "dist_type": "surrounded_and_reflect",
          "p": 0.5,
          "map_x": 32,
          "map_y": 32,
      }
  }
  if 'protoss' in map_name:
    distribution_config['team_gen'] = {
        "dist_type": "weighted_teams",
        "unit_types": ["stalker", "zealot", "colossus"],
        "weights": [0.45, 0.45, 0.1],
        "observe": True,
    }
  elif 'zerg' in map_name:
    distribution_config['team_gen'] = {
        "dist_type": "weighted_teams",
        "unit_types": ["zergling", "baneling", "hydralisk"],
        "weights": [0.45, 0.1, 0.45],
        "observe": True,
    }
  elif 'terran' in map_name:
    distribution_config['team_gen'] = {
        "dist_type": "weighted_teams",
        "unit_types": ["marine", "marauder", "medivac"],
        "weights": [0.45, 0.45, 0.1],
        "observe": True,
    }
  return distribution_config


class SMAC_V2(ParallelEnv):
  metadata = {"render_modes": ["human"], "name": "smac_v2"}

  def __init__(self, n_allies, n_enemies, map_name, max_length):
    capability_config = parse_smacv2_distribution(n_allies, n_enemies, map_name)
    self.env = SMACv2(capability_config=capability_config, map_name=map_name)

    self.max_length = max_length
    self.cur_step = 0
    self.possible_agents = list(range(self.env.n_agents))
    self.agents = self.possible_agents

    self.observation_spaces = []
    for size in self.env.observation_space:
      self.observation_spaces.append(
          Box(low=-np.ones(size[0]), high=np.ones(size[0])))
    self.action_spaces = []
    for aspace in self.env.action_space:
      self.action_spaces.append(Discrete(aspace.n))

  @functools.lru_cache(maxsize=None)
  def observation_space(self, agent):
    # gymnasium spaces are defined and documented here:
    #    https://gymnasium.farama.org/api/spaces/
    return self.observation_spaces[agent]

  @functools.lru_cache(maxsize=None)
  def action_space(self, agent):
    return self.action_spaces[agent]

  def close(self):
    pass

  def reset(self, seed=None, options=None):
    if seed is not None:
      self.env.seed(seed)

    self.cur_step = 0
    obs, state, avail_actions = self.env.reset()

    infos = []
    for idx in range(self.env.n_agents):
      infos.append({"avail_actions": np.array(avail_actions[idx])})

    return np.array(obs), infos

  def step(self, actions):
    list_actions = []
    for aname in self.agents:
      list_actions.append(actions[aname])

    obs, state, rewards, dones, infos, avail_actions = self.env.step(
        np.array(list_actions))
    self.cur_step += 1

    trunc = self.cur_step >= self.max_length

    if np.all(dones):
      dones = {agent: True for agent in self.agents}
      truncs = {agent: False for agent in self.agents}
    else:
      dones = {agent: False for agent in self.agents}
      truncs = {agent: trunc for agent in self.agents}

    new_infos = {aname: {} for aname in self.agents}
    new_infos = {}
    for aname in self.agents:
      new_infos[aname] = {}
      for key, val in infos[aname].items():
        new_infos[aname][key] = val

      new_infos[aname]["avail_actions"] = avail_actions[aname]

    return np.array(obs), np.array(rewards).reshape(
        -1), dones, truncs, new_infos

  def save_replay(self):
    self.env.save_replay()


class Protoss5v5(SMAC_V2):

  def __init__(self):
    super().__init__(5, 5, '10gen_protoss', 200)


class Terran5v5(SMAC_V2):

  def __init__(self):
    super().__init__(5, 5, '10gen_terran', 200)


class MAPPO_Expert:

  def __init__(self, model_dir, obs_space, act_space) -> None:
    args = DictConfig({})
    args["hidden_size"] = 64
    args["gain"] = 0.01
    args["use_orthogonal"] = True
    args["use_policy_active_masks"] = True
    args["algorithm_name"] = "mappo"
    args["use_recurrent_policy"] = False
    args["use_naive_recurrent_policy"] = False
    args["recurrent_N"] = 1
    args["use_feature_normalization"] = True
    args["use_ReLU"] = True
    args["stacked_frames"] = 1
    args["layer_N"] = 1
    device = "cpu"

    self.actor = R_Actor(args, obs_space, act_space, device)

    # Restore policy's networks from a saved model.
    policy_actor_state_dict = torch.load(str(model_dir) + '/actor.pt')
    self.actor.load_state_dict(policy_actor_state_dict)

    self.actor.eval()

  def choose_action(self, obs, available_actions):
    dummy_value = np.zeros(2)
    deterministic = True
    actions, _, _ = self.actor(np.array(obs), dummy_value, dummy_value,
                               available_actions, deterministic)
    return actions


class Protoss_Expert(MAPPO_Expert):

  def __init__(self, obs_space, act_space):
    model_dir = (
        "/home/sangwon/Projects/ai_coach/domains/pettingzoo_domain/models/"
        "StarCraft2v2/10gen_protoss/mappo/protoss/run1/models")
    super().__init__(model_dir, obs_space, act_space)


class Terran_Expert(MAPPO_Expert):

  def __init__(self, obs_space, act_space):
    model_dir = (
        "/home/sangwon/Projects/ai_coach/domains/pettingzoo_domain/models/"
        "StarCraft2v2/10gen_terran/mappo/terran/run1/models")
    super().__init__(model_dir, obs_space, act_space)


def generate_data(save_dir, env_name, n_traj):
  if env_name == "Protoss5v5":
    env = Protoss5v5()
    agent = Protoss_Expert(env.observation_space(0), env.action_space(0))
  elif env_name == "Terran5v5":
    env = Terran5v5()
    agent = Terran_Expert(env.observation_space(0), env.action_space(0))

  num_win = 0

  expert_trajs = defaultdict(list)
  list_total_reward = []
  for _ in range(n_traj):
    obs, infos = env.reset(0)
    episode_reward = {aname: 0 for aname in env.agents}

    is_won = False
    total_reward = 0
    samples = []
    while True:
      actions = {}
      # Sample actions
      with torch.no_grad():
        available_actions = np.array(
            [infos[aname]["avail_actions"] for aname in env.agents])
        actions = agent.choose_action(obs, available_actions)

      # Obser reward and next obs
      next_obs, rewards, dones, truncs, infos = env.step(actions)

      samples.append((np.array(obs), np.array(available_actions),
                      np.array(actions), np.array(next_obs), rewards, dones))

      obs = next_obs

      for aname in env.agents:
        episode_reward[aname] += rewards[aname]
        total_reward += rewards[aname]

      if all(truncs.values()) or all(dones.values()):
        if infos[0]['won']:
          num_win += 1
          is_won = True
        break

    list_total_reward.append(total_reward)

    # structure: (traj_length, num_agents, value_dim)
    (all_obs, all_avail_act, all_act, all_n_obs, all_rew,
     all_don) = list(zip(*samples))

    def transpose_impl(all_item):
      return [[item[aname] for item in all_item] for aname in env.agents]

    expert_trajs["states"].append(transpose_impl(all_obs))
    expert_trajs["avail_actions"].append(transpose_impl(all_avail_act))
    expert_trajs["next_states"].append(transpose_impl(all_n_obs))
    expert_trajs["actions"].append(transpose_impl(all_act))
    expert_trajs["rewards"].append(transpose_impl(all_rew))
    expert_trajs["dones"].append(transpose_impl(all_don))
    expert_trajs["wons"].append(int(is_won))
    expert_trajs["lengths"].append(len(all_obs))

  print("Mean:", np.mean(list_total_reward))
  print("win rate:", np.mean(expert_trajs["wons"]))
  if save_dir is not None:
    file_path = os.path.join(save_dir, f"{env_name}_{n_traj}.pkl")
    with open(file_path, 'wb') as f:
      pickle.dump(expert_trajs, f)

  return expert_trajs


if __name__ == "__main__":
  cur_dir = os.path.dirname(__file__)

  # traj = generate_data(cur_dir, LDExpert, "MultiJobs2", 100, False, 300)
  # traj = generate_data(None, LDExpert, "MultiJobs2", 10, False, 100)
  # traj = generate_data(cur_dir, "Protoss5v5", 100)
  traj = generate_data(cur_dir, "Terran5v5", 100)

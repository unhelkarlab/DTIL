import os
import torch
import numpy as np
from aic_ml.baselines.IQLearn.dataset.expert_dataset import read_file
from aic_ml.baselines.IQLearn.utils.utils import eval_mode, one_hot
from pettingzoo.utils.env import ParallelEnv


def conv_input(batch_input, is_onehot_needed, dimension, device):
  if is_onehot_needed:
    if not isinstance(batch_input, torch.Tensor):
      batch_input = torch.tensor(np.array(batch_input),
                                 dtype=torch.float).reshape(-1).to(device)
    else:
      batch_input = batch_input.reshape(-1)
    # TODO: used a trick to handle initial/unobserved values.
    #       may need to find a better way later
    batch_input = one_hot(batch_input, dimension + 1)
    batch_input = batch_input[:, :-1]
  else:
    if not isinstance(batch_input, torch.Tensor):
      batch_input = torch.tensor(np.array(batch_input).reshape(-1, dimension),
                                 dtype=torch.float).to(device)
    else:
      batch_input = batch_input.reshape(-1, dimension).to(device)

  return batch_input  # TODO: consider to(device) here too. anything to lose?


def conv_tuple_input(tup_batch, tup_is_onehot_needed, tup_dimension, device):
  list_batch = []
  for idx in range(len(tup_batch)):
    batch = conv_input(tup_batch[idx], tup_is_onehot_needed[idx],
                       tup_dimension[idx], device)
    list_batch.append(batch)

  # concat
  batch_input = torch.cat(list_batch, dim=1)

  return batch_input


def split_by_size(batch_input, split_size, device):
  if len(split_size) == 0:
    return ()
  else:
    return torch.split(torch.as_tensor(batch_input, device=device),
                       split_size,
                       dim=-1)


def get_expert_batch(expert_traj,
                     mental_states,
                     device,
                     init_latent,
                     mental_states_after_end=None,
                     has_mental=True):
  '''
  if has_mental is False, all mental-related input arguments will be ignored
          and "prev_latents", "latents" and "next_latents" will not be returned.
  return: dictionary with these keys: states, prev_latents, prev_auxs, 
                            next_states, latents, actions, auxs, rewards, dones
  '''
  num_samples = len(expert_traj["states"])

  dict_batch = {}
  dict_batch['states'] = []
  dict_batch['prev_auxs'] = []
  dict_batch['next_states'] = []
  dict_batch['actions'] = []
  dict_batch['auxs'] = []
  dict_batch['rewards'] = []
  dict_batch['dones'] = []

  if has_mental:
    dict_batch['prev_latents'] = []
    dict_batch['latents'] = []
    if mental_states_after_end is not None:
      dict_batch['next_latents'] = []

    init_latent = np.array(init_latent).reshape(-1)

  for i_e in range(num_samples):
    length = len(expert_traj["rewards"][i_e])

    dict_batch['states'].append(
        np.array(expert_traj["states"][i_e]).reshape(length, -1))
    dict_batch['prev_auxs'].append(
        np.array(expert_traj["prev_auxs"][i_e]).reshape(length, -1))
    dict_batch['next_states'].append(
        np.array(expert_traj["next_states"][i_e]).reshape(length, -1))
    dict_batch['actions'].append(
        np.array(expert_traj["actions"][i_e]).reshape(length, -1))
    dict_batch['auxs'].append(
        np.array(expert_traj["auxs"][i_e]).reshape(length, -1))
    dict_batch['rewards'].append(
        np.array(expert_traj["rewards"][i_e]).reshape(-1, 1))
    dict_batch['dones'].append(
        np.array(expert_traj["dones"][i_e]).reshape(-1, 1))

    if has_mental:
      dict_batch['prev_latents'].append(init_latent)
      dict_batch['prev_latents'].append(
          np.array(mental_states[i_e][:-1]).reshape(-1, 1))
      dict_batch['latents'].append(np.array(mental_states[i_e]).reshape(-1, 1))

      if mental_states_after_end is not None:
        dict_batch["next_latents"].append(
            np.array(mental_states[i_e][1:]).reshape(-1, 1))
        dict_batch["next_latents"].append(
            np.array(mental_states_after_end[i_e]).reshape(-1))

  for key, val in dict_batch.items():
    tmp = np.vstack(val)
    dict_batch[key] = torch.as_tensor(tmp, dtype=torch.float, device=device)

  return dict_batch


def get_samples(batch_size, dataset):
  indexes = np.random.choice(np.arange(len(dataset['states'])),
                             size=batch_size,
                             replace=False)

  batch = {}
  for key, val in dataset.items():
    batch[key] = val[indexes]

  return batch


def conv_samples_tup2dict(tup_batch, dict_keys):
  batch = {}
  for idx in range(len(tup_batch)):
    batch[dict_keys[idx]] = tup_batch[idx]

  return batch


def get_concat_samples(policy_batch, expert_batch, is_sqil: bool = False):
  '''
  policy_batch, expert_batch: the 2nd last item should be reward,
                                and the last item should be done
  return: concatenated batch with an additional item of is_expert
  '''

  concat_batch = {}
  for key in policy_batch.keys():
    if key == "rewards":
      # ----- concat reward data
      online_batch_reward = policy_batch[key]
      expert_batch_reward = expert_batch[key]
      if is_sqil:
        # convert policy reward to 0
        online_batch_reward = torch.zeros_like(online_batch_reward)
        # convert expert reward to 1
        expert_batch_reward = torch.ones_like(expert_batch_reward)
      concat_batch[key] = torch.cat([online_batch_reward, expert_batch_reward],
                                    dim=0)
    else:
      concat_batch[key] = torch.cat([policy_batch[key], expert_batch[key]],
                                    dim=0)

  concat_batch["is_expert"] = torch.cat([
      torch.zeros_like(online_batch_reward, dtype=torch.bool),
      torch.ones_like(expert_batch_reward, dtype=torch.bool)
  ],
                                        dim=0)

  return concat_batch


def load_trajectories(expert_location: str,
                      num_trajectories: int = 10,
                      seed: int = 0):
  """Load expert trajectories
    Assumes expert dataset is a dict with keys {states, actions, ...}
    with values of given shapes below:
        expert["states"]  =  [num_trajs, num_agents, traj_length, state_space]
        expert["actions"] =  [num_trajs, num_agents, traj_length, action_space]
        expert["rewards"] =  [num_trajs, num_agents, traj_length]
        expert["lengths"] =  [num_trajs]

        (optional)
        expert["auxs"]    =  [num_trajs, num_agents, traj_length, aux_space]
        expert["latents"] =  [num_trajs, num_agents, traj_length, latent_space]

    Args:
        expert_location:          Location of saved expert trajectories.
        num_trajectories:         Number of trajectories to sample (randomized).

    Returns:
        List of Dict. Each Dict corresponds to an agent.
    """
  if os.path.isfile(expert_location):
    # Load data from single file.
    with open(expert_location, 'rb') as f:
      trajs = read_file(expert_location, f)

    rng = np.random.RandomState(seed)
    # Sample random `num_trajectories` experts.
    perm = np.arange(len(trajs["states"]))
    perm = rng.permutation(perm)

    idx = perm[:num_trajectories]
    for k, v in trajs.items():
      # if not torch.is_tensor(v):
      #     v = np.array(v)  # convert to numpy array
      trajs[k] = [v[i] for i in idx]

  else:
    raise ValueError(f"{expert_location} is not a valid path")

  num_agents = len(trajs["states"][0])
  num_trajs = len(trajs["states"])

  list_each_agent = []
  for i_agent in range(num_agents):
    each_agent = {}
    for k, v in trajs.items():
      if k == "lengths" or k == "wons":
        each_agent[k] = v
      else:
        each_agent[k] = []
        for i_epi in range(num_trajs):
          each_agent[k].append(v[i_epi][i_agent])

    list_each_agent.append(each_agent)

  return list_each_agent


def save(dict_agents, env_name, output_dir='results', suffix=""):
  if not os.path.exists(output_dir):
    os.mkdir(output_dir)

  for a_name in dict_agents:
    file_path = os.path.join(output_dir, f'{env_name}' + suffix + f'_{a_name}')
    dict_agents[a_name].save(file_path)


def evaluate(dict_agents, env: ParallelEnv, use_auxiliary_obs, num_episodes=10):
  total_timesteps = []
  total_returns = {a_name: [] for a_name in env.agents}
  wins = []

  while len(total_timesteps) < num_episodes:
    done = False
    is_win = False
    episode_rewards = {a_name: 0 for a_name in env.agents}
    episode_step = 0

    joint_obs, infos = env.reset()
    joint_prev_lat = {}
    joint_prev_aux = {}
    for a_name in env.agents:
      agent = dict_agents[a_name]
      prev_lat, prev_aux = agent.PREV_LATENT, agent.PREV_AUX
      joint_prev_lat[a_name] = prev_lat
      joint_prev_aux[a_name] = prev_aux

    while not done:
      joint_latent = {}
      joint_actions = {}
      joint_aux = {}
      for a_name in env.agents:
        agent = dict_agents[a_name]
        if "avail_actions" in infos[a_name]:
          available_actions = np.array(infos[a_name]["avail_actions"])
        else:
          available_actions = None
        latent, action = agent.choose_action(joint_obs[a_name],
                                             joint_prev_lat[a_name],
                                             joint_prev_aux[a_name],
                                             sample=False,
                                             avail_actions=available_actions)
        joint_latent[a_name] = latent
        joint_actions[a_name] = action

      joint_next_obs, rewards, dones, truncates, infos = env.step(joint_actions)
      episode_step += 1
      for a_name in env.agents:
        if use_auxiliary_obs:
          joint_aux[a_name] = env.get_auxiliary_obs(a_name)
        else:
          joint_aux[a_name] = agent.PREV_AUX

        episode_rewards[a_name] += rewards[a_name]

        if dones[a_name] or truncates[a_name]:
          done = True

        if "won" in infos[a_name] and infos[a_name]["won"]:
          is_win = True

      joint_obs = joint_next_obs
      joint_prev_lat = joint_latent
      joint_prev_aux = joint_aux

    for a_name in env.agents:
      total_returns[a_name].append(episode_rewards[a_name])
    total_timesteps.append(episode_step)
    wins.append(int(is_win))

  return total_returns, total_timesteps, wins

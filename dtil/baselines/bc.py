from typing import Dict, Any, Sequence
import os
import numpy as np
import math
import random
# import torch.optim as optim
# import torch.nn as nn
import torch
from torch import Tensor
from torch.utils.tensorboard import SummaryWriter
import wandb
import omegaconf
from gymnasium.spaces import Discrete, Box
from gym.spaces import Discrete as GymDiscrete
from gym.spaces import Box as GymBox
from .IQLearn.utils.logger import Logger
from .IQLearn.agent.sac_models import AbstractActor
from .IQLearn.agent.sac_models import DiscreteActor, DiagGaussianActor
from aic_ml.MAHIL.helper.utils import conv_input, load_trajectories
from pettingzoo.utils.env import ParallelEnv  # noqa: F401


class BehaviorCloning:

  def __init__(self, config: omegaconf.DictConfig, policy: AbstractActor,
               dim_obs, discrete_obs, dim_act, discrete_act, device) -> None:

    self.policy = policy
    self.device = device

    self.dim_obs = dim_obs
    self.discrete_obs = discrete_obs
    self.dim_act = dim_act
    self.discrete_act = discrete_act

    self.optimizer = torch.optim.Adam(policy.parameters(),
                                      lr=config.optimizer_lr_policy,
                                      betas=[0.9, 0.999])

    if discrete_act:
      self.criterion = torch.nn.CrossEntropyLoss()
    else:
      self.criterion = torch.nn.MSELoss()

    self.policy.train()

  def train_one_batch(self, batch_obs_action):
    obs, action = list(zip(*batch_obs_action))
    nn_input = conv_input(obs, self.discrete_obs, self.dim_obs, self.device)

    dist = self.policy.forward(nn_input)
    if self.discrete_act:
      target = conv_input(action, False, 1, self.device).long()
      loss = self.criterion(dist.logits, target.reshape(-1))  # type: Tensor
    else:
      target = conv_input(action, self.discrete_act, self.dim_act, self.device)
      loss = self.criterion(dist.mean, target)  # type: Tensor

    self.optimizer.zero_grad()
    loss.backward()
    self.optimizer.step()

    return loss


def get_obs_act_space_info(env: ParallelEnv, agent_idx):

  agent_name = env.agents[agent_idx]
  obs_space = env.observation_space(agent_name)

  if isinstance(obs_space, Discrete) or isinstance(obs_space, GymDiscrete):
    dim_obs = obs_space.n
    discrete_obs = True
  elif isinstance(obs_space, Box) or isinstance(obs_space, GymBox):
    dim_obs = obs_space.shape[0]
    discrete_obs = False
  else:
    raise ValueError("Unsupported observation space")

  act_space = env.action_space(agent_name)
  if isinstance(act_space, Discrete) or isinstance(act_space, GymDiscrete):
    dim_act = act_space.n
    discrete_act = True
  elif isinstance(act_space, Box) or isinstance(act_space, GymBox):
    dim_act = act_space.shape[0]
    discrete_act = False
  else:
    raise ValueError("Unsupported action space")

  return dim_obs, discrete_obs, dim_act, discrete_act


def eval_policy(env: ParallelEnv, list_policy: Sequence[AbstractActor],
                num_episodes, device):
  n_agents = len(list_policy)
  total_timesteps = []
  total_returns = {a_name: [] for a_name in env.agents}
  wins = []

  while len(total_timesteps) < num_episodes:
    done = False
    is_win = False
    episode_rewards = {a_name: 0 for a_name in env.agents}
    episode_step = 0

    joint_obs, infos = env.reset()
    while not done:
      joint_actions = {}
      for a_idx, aname in enumerate(env.agents):
        agent = list_policy[a_idx]
        if "avail_actions" in infos[aname]:
          available_actions = np.array(infos[aname]["avail_actions"])
        else:
          available_actions = None

        dim_obs, discrete_obs, _, _ = get_obs_act_space_info(env, a_idx)

        with torch.no_grad():
          nn_obs = conv_input(joint_obs[aname], discrete_obs, dim_obs, device)
          if isinstance(agent, DiscreteActor):
            action = agent.sample_w_avail_actions(nn_obs, available_actions)
          else:
            action = agent.exploit(nn_obs)
          action = action.cpu().numpy()[0]

        joint_actions[aname] = action

      joint_obs, rewards, dones, truncates, infos = env.step(joint_actions)

      episode_step += 1
      for a_name in env.agents:
        episode_rewards[a_name] += rewards[a_name]

        if dones[a_name] or truncates[a_name]:
          done = True

        if "won" in infos[a_name] and infos[a_name]["won"]:
          is_win = True

    for a_name in env.agents:
      total_returns[a_name].append(episode_rewards[a_name])
    total_timesteps.append(episode_step)
    wins.append(int(is_win))

  return total_returns, total_timesteps, wins


def make_actor(config: omegaconf.DictConfig, dim_obs, discrete_obs, dim_act,
               discrete_act, device):

  if discrete_act:
    actor = DiscreteActor(dim_obs, dim_act, config.hidden_policy).to(device)
  else:
    actor = DiagGaussianActor(
        dim_obs,
        dim_act,
        config.hidden_policy,
        config.log_std_bounds,
        config.bounded,
        use_nn_logstd=config.use_nn_logstd,
        clamp_action_logstd=config.clamp_action_logstd).to(device)

  return actor


def train_bc(config, demo_path, log_dir, output_dir, cb_env_factory,
             log_interval, eval_interval, env_kwargs):
  env_name = config.env_name
  seed = config.seed
  num_episodes = config.num_eval_episodes
  num_trajs = config.n_traj

  device = torch.device(config.device)

  dict_config = omegaconf.OmegaConf.to_container(config,
                                                 resolve=True,
                                                 throw_on_missing=True)

  alg_name = config.alg_name
  run_name = f"{alg_name}_{config.tag}"
  wandb.init(project=env_name,
             name=run_name,
             entity='sangwon-seo',
             sync_tensorboard=True,
             reinit=True,
             config=dict_config)

  # set seeds
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)

  if cb_env_factory is not None:
    env = cb_env_factory(**env_kwargs)  # type: ParallelEnv

    # Seed envs
    env.reset(seed=seed)
    n_agents = env.num_agents
  else:
    env = None
    n_agents = config.n_agents

  # ----- create agents
  list_agents = []  # type: Sequence[AbstractActor]
  list_bc_learners = []  # type: Sequence[BehaviorCloning]
  for agent_idx in range(n_agents):
    if env is None:
      dim_obs = config.dim_obs[agent_idx]
      discrete_obs = config.discrete_obs[agent_idx]
      dim_act = config.dim_act[agent_idx]
      discrete_act = config.discrete_act[agent_idx]
    else:
      dim_obs, discrete_obs, dim_act, discrete_act = get_obs_act_space_info(
          env, agent_idx)

    agent = make_actor(config, dim_obs, discrete_obs, dim_act, discrete_act,
                       device)
    bc_learner = BehaviorCloning(config, agent, dim_obs, discrete_obs, dim_act,
                                 discrete_act, device)

    list_agents.append(agent)
    list_bc_learners.append(bc_learner)

  # ----- Load expert data
  list_expert_trajs = load_trajectories(demo_path, num_trajs, seed + 42)
  num_trajs = len(list_expert_trajs[0]["lengths"])
  for agent_idx in range(n_agents):
    expert_trajs = list_expert_trajs[agent_idx]
    list_s_a_pairs = []
    for i_e in range(num_trajs):
      obs = expert_trajs["states"][i_e]
      acts = expert_trajs["actions"][i_e]
      s_a_pairs = list(zip(obs, acts))
      list_s_a_pairs.extend(s_a_pairs)

    list_expert_trajs[agent_idx] = list_s_a_pairs

  num_samples = len(list_expert_trajs[0])

  # Setup logging
  writer = SummaryWriter(log_dir=log_dir)
  print(f'--> Saving logs at: {log_dir}')
  logger = Logger(log_dir,
                  log_frequency=log_interval,
                  writer=writer,
                  save_tb=True,
                  run_name=f"{env_name}_{run_name}")

  # ----- Train BC
  batch_sz = config.mini_batch_size
  n_batch_per_epoch = math.ceil(num_samples / batch_sz)
  best_reward = -np.inf
  for learning_step in range(config.n_batches):
    i_b = learning_step % n_batch_per_epoch
    if i_b == 0:
      for i_a in range(n_agents):
        random.shuffle(list_expert_trajs[i_a])

    i_log = learning_step % log_interval
    start = i_b * batch_sz
    end = min((i_b + 1) * batch_sz, num_samples)
    for i_a in range(n_agents):
      batch = list_expert_trajs[i_a][start:end]
      loss = list_bc_learners[i_a].train_one_batch(batch)

      if i_log == 0:
        writer.add_scalar(f"train/loss/{i_a}", loss, global_step=learning_step)

    if env is not None and learning_step % eval_interval == 0:
      # Evaluate
      dict_eval_returns, eval_timesteps, wins = eval_policy(
          env, list_agents, num_episodes, device)
      ret_sum = np.zeros_like(dict_eval_returns[env.agents[0]])
      for a_name in env.agents:
        ret_sum = ret_sum + np.array(dict_eval_returns[a_name])
        logger.log(f'eval/returns/{a_name}', np.mean(dict_eval_returns[a_name]),
                   learning_step)

      mean_ret_sum = np.mean(ret_sum)
      logger.log('eval/episode_reward', mean_ret_sum, learning_step)
      logger.log('eval/episode_step', np.mean(eval_timesteps), learning_step)
      logger.log('eval/win_rate', np.mean(wins), learning_step)
      logger.dump(learning_step, ty='eval')

      if mean_ret_sum >= best_reward:
        # Store best eval returns
        best_reward = mean_ret_sum
        wandb.run.summary["best_returns"] = best_reward

        for a_idx in range(len(list_agents)):
          file_path = os.path.join(output_dir,
                                   f'{env_name}_n{num_trajs}_l0_best_{a_idx}')
          torch.save(list_agents[a_idx].state_dict(), file_path)

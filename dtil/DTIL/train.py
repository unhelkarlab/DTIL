from typing import Sequence, Dict, Any
import random
import numpy as np
import torch
from itertools import count
from torch.utils.tensorboard import SummaryWriter
from collections import deque
from ..helper.utils import eval_mode
from ..helper.logger import Logger
from ..helper.option_memory import OptionMemory
from ..helper.utils import (load_multiagent_data_w_labels,
                            infer_mental_states_all_demo,
                            infer_last_next_mental_state, get_expert_batch,
                            get_samples, conv_samples_tup2dict, save, evaluate)
from .agent import MAHIL, make_mahil_agent
import wandb
import omegaconf
from pettingzoo.utils.env import ParallelEnv  # noqa: F401


def train(config: omegaconf.DictConfig,
          demo_path,
          log_dir,
          output_dir,
          cb_env_factory,
          log_interval=500,
          eval_interval=5000,
          env_kwargs={},
          cb_ex_eval=None):

  env_name = config.env_name
  seed = config.seed
  batch_size = config.mini_batch_size
  max_explore_step = config.max_explore_step
  eps_window = 10
  num_episodes = config.num_eval_episodes
  num_trajs = config.n_traj

  dict_config = omegaconf.OmegaConf.to_container(config,
                                                 resolve=True,
                                                 throw_on_missing=True)

  run_name = f"{config.alg_name}_{config.tag}"
  wandb.init(project=env_name,
             name=run_name,
             entity='sangwon-seo',
             sync_tensorboard=True,
             reinit=True,
             config=dict_config)

  # device
  # device = torch.device(config.device)

  # set seeds
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)

  env = cb_env_factory(**env_kwargs)  # type: ParallelEnv
  eval_env = cb_env_factory(**env_kwargs)  # type: ParallelEnv

  # Seed envs
  env.reset(seed=seed)
  eval_env.reset(seed=seed + 10)

  n_init_mem = int(config.init_sample)
  n_replay_mem = int(config.n_sample)
  assert n_init_mem <= n_replay_mem
  eps_window = int(eps_window)
  max_explore_step = int(max_explore_step)

  # ----- create agents
  dict_agents = {}  # type: Dict[Any, MAHIL]
  dict_replay_memory = {}  # type: Dict[Any, OptionMemory]
  for agent_idx in range(env.num_agents):
    agent = make_mahil_agent(config, env, agent_idx)
    a_name = env.agents[agent_idx]
    dict_agents[a_name] = agent
    dict_replay_memory[a_name] = OptionMemory(n_replay_mem, seed + 1)

  # ----- Load expert data
  n_labeled = int(num_trajs * config.supervision)
  dict_expert_trajs, dict_expert_labels, cnt_label, n_expert_samples = (
      load_multiagent_data_w_labels(env.agents, dict_agents, demo_path,
                                    num_trajs, n_labeled, seed))

  # expert_avg, expert_std = compute_expert_return_mean(
  #     list_expert_trajs)

  # wandb.run.summary["expert_avg"] = expert_avg
  # wandb.run.summary["expert_std"] = expert_std

  output_suffix = f"_n{num_trajs}_l{cnt_label}"
  batch_size = min(batch_size, n_expert_samples)

  # Setup logging
  writer = SummaryWriter(log_dir=log_dir)
  print(f'--> Saving logs at: {log_dir}')
  logger = Logger(log_dir,
                  log_frequency=log_interval,
                  writer=writer,
                  save_tb=True,
                  run_name=f"{env_name}_{run_name}")

  # track mean reward and scores
  best_eval_returns = -np.inf
  dict_rewards_window = {
      a_name: deque(maxlen=eps_window)
      for a_name in env.agents
  }  # last N rewards
  epi_step_window = deque(maxlen=eps_window)
  epi_win_window = deque(maxlen=eps_window)
  cnt_steps = 0

  begin_learn = False
  explore_steps = 0
  dict_expert_data = {}

  for epoch in count():
    episode_rewards = {a_name: 0 for a_name in env.agents}
    env_done = False
    is_win = False

    joint_obs, infos = env.reset()
    joint_prev_lat = {}
    joint_prev_aux = {}
    joint_latent = {}
    for a_name in env.agents:
      agent = dict_agents[a_name]
      prev_lat, prev_aux = agent.PREV_LATENT, agent.PREV_AUX
      latent = agent.choose_mental_state(joint_obs[a_name],
                                         prev_lat,
                                         prev_aux,
                                         sample=True)
      joint_prev_lat[a_name] = prev_lat
      joint_prev_aux[a_name] = prev_aux
      joint_latent[a_name] = latent

    for episode_step in count():
      joint_actions = {}
      for a_name in env.agents:
        agent = dict_agents[a_name]
        with eval_mode(agent):
          if "avail_actions" in infos[a_name]:
            available_actions = np.array(infos[a_name]["avail_actions"])
          else:
            available_actions = None

          action = agent.choose_policy_action(joint_obs[a_name],
                                              joint_latent[a_name],
                                              sample=True,
                                              avail_actions=available_actions)
          joint_actions[a_name] = action

      joint_next_obs, rewards, dones, truncates, infos = env.step(joint_actions)
      joint_aux = {}
      joint_next_latent = {}
      for a_name in env.agents:
        agent = dict_agents[a_name]

        if config.use_auxiliary_obs:
          # TODO: implement get_aux at custom methods
          joint_aux[a_name] = env.get_auxiliary_obs(a_name)
        else:
          joint_aux[a_name] = agent.PREV_AUX

        next_latent = agent.choose_mental_state(joint_next_obs[a_name],
                                                joint_latent[a_name],
                                                joint_aux[a_name],
                                                sample=True)
        joint_next_latent[a_name] = next_latent

        episode_rewards[a_name] += rewards[a_name]

        dict_replay_memory[a_name].add(
            (joint_prev_lat[a_name], joint_prev_aux[a_name], joint_obs[a_name],
             joint_latent[a_name], joint_actions[a_name], joint_aux[a_name],
             joint_next_obs[a_name], joint_next_latent[a_name], rewards[a_name],
             dones[a_name]))

        if dones[a_name] or truncates[a_name]:
          env_done = True

      if explore_steps % eval_interval == 0 and begin_learn:
        dict_eval_returns, eval_timesteps, wins = evaluate(
            dict_agents,
            eval_env,
            config.use_auxiliary_obs,
            num_episodes=num_episodes)
        ret_sum = np.zeros_like(dict_eval_returns[env.agents[0]])
        for a_name in env.agents:
          ret_sum = ret_sum + np.array(dict_eval_returns[a_name])
          logger.log(f'eval/returns/{a_name}',
                     np.mean(dict_eval_returns[a_name]), explore_steps)

        mean_ret_sum = np.mean(ret_sum)
        logger.log('eval/episode_reward', mean_ret_sum, explore_steps)
        logger.log('eval/episode_step', np.mean(eval_timesteps), explore_steps)
        logger.log('eval/win_rate', np.mean(wins), explore_steps)

        if cb_ex_eval is not None:
          dict_ex_metrics = cb_ex_eval(dict_agents, env.agents)
          for key, val in dict_ex_metrics.items():
            logger.log(f'eval/{key}', val, explore_steps)

        logger.dump(explore_steps, ty='eval')

        if mean_ret_sum >= best_eval_returns:
          # Store best eval returns
          best_eval_returns = mean_ret_sum
          wandb.run.summary["best_returns"] = best_eval_returns
          save(dict_agents,
               env_name,
               output_dir=output_dir,
               suffix=output_suffix + "_best")

      explore_steps += 1
      if dict_replay_memory[env.agents[0]].size() >= n_init_mem:
        # Start learning
        if not begin_learn:
          print('Learn begins!')
          begin_learn = True

        if explore_steps == max_explore_step:
          print('Finished!')
          wandb.finish()
          return

        # ##### sample batch
        # infer mental states of expert data
        if (len(dict_expert_data) == 0
            or explore_steps % config.demo_latent_infer_interval == 0):
          for a_name in env.agents:
            agent = dict_agents[a_name]
            expert_traj = dict_expert_trajs[a_name]
            mental_states = infer_mental_states_all_demo(
                agent, expert_traj, dict_expert_labels[a_name])
            end_mental_state = infer_last_next_mental_state(
                agent, expert_traj, mental_states)
            dict_expert_data[a_name] = get_expert_batch(expert_traj,
                                                        mental_states,
                                                        agent.device,
                                                        agent.PREV_LATENT,
                                                        end_mental_state)

        ######
        # IQ-Learn
        dict_tx_losses = dict_pi_losses = {a_name: {} for a_name in env.agents}
        if explore_steps % config.update_interval == 0:
          for a_name in env.agents:
            agent = dict_agents[a_name]
            policy_tup_batch = dict_replay_memory[a_name].get_samples(
                batch_size, agent.device)
            policy_batch = conv_samples_tup2dict(policy_tup_batch, [
                'prev_latents', 'prev_auxs', 'states', 'latents', 'actions',
                'auxs', 'next_states', 'next_latents', 'rewards', 'dones'
            ])
            expert_batch = get_samples(batch_size, dict_expert_data[a_name])

            dict_tx_losses[a_name], dict_pi_losses[a_name] = agent.mahil_update(
                policy_batch, expert_batch, config.demo_latent_infer_interval,
                logger, explore_steps)

        if explore_steps % log_interval == 0:
          for a_name in env.agents:
            for key, loss in dict_tx_losses[a_name].items():
              writer.add_scalar(f"tx_loss/{a_name}/" + key,
                                loss,
                                global_step=explore_steps)
            for key, loss in dict_pi_losses[a_name].items():
              writer.add_scalar(f"pi_loss/{a_name}/" + key,
                                loss,
                                global_step=explore_steps)

      if env_done:
        if 'won' in infos[env.agents[0]]:
          is_win = infos[env.agents[0]]['won']
        break

      joint_obs = joint_next_obs
      joint_prev_lat = joint_latent
      joint_prev_aux = joint_aux
      joint_latent = joint_next_latent

    for a_name in env.agents:
      dict_rewards_window[a_name].append(episode_rewards[a_name])
    epi_step_window.append(episode_step + 1)
    epi_win_window.append(int(is_win))
    cnt_steps += episode_step + 1
    if cnt_steps >= log_interval:
      cnt_steps = 0
      logger.log('train/episode', epoch, explore_steps)
      logger.log('train/episode_step', np.mean(epi_step_window), explore_steps)
      logger.log('train/win_rate', np.mean(epi_win_window), explore_steps)

      ret_sum = np.zeros_like(dict_rewards_window[env.agents[0]])
      for a_name in env.agents:
        ret_sum = ret_sum + np.array(dict_rewards_window[a_name])
        logger.log(f'train/returns/{a_name}',
                   np.mean(dict_rewards_window[a_name]), explore_steps)
      logger.log('train/episode_reward', np.mean(ret_sum), explore_steps)

      logger.dump(explore_steps, save=begin_learn)

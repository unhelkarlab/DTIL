import os
import torch
import random
import numpy as np
from collections import defaultdict
from typing import Dict, Any
from itertools import count
from .utils.logger import Logger
import wandb
import omegaconf
from pettingzoo.utils.env import ParallelEnv
from ...helper.utils import (load_multiagent_data_w_labels,
                             infer_mental_states_all_demo, evaluate)
from .model.agent import make_agent, MA_OGAIL


def learn(config: omegaconf.DictConfig,
          use_option,
          demo_path,
          log_dir,
          save_dir,
          cb_env_factory,
          pretrain_name,
          eval_interval,
          env_kwargs={}):

  n_traj = config.n_traj
  n_sample = config.n_sample
  max_exp_step = config.max_explore_step
  seed = config.seed
  env_name = config.env_name
  num_episodes = config.num_eval_episodes

  # env_type = config.env_type
  # use_pretrain = config.use_pretrain
  # n_thread = config.n_thread
  # n_iter = config.n_pretrain_epoch
  # log_interval = config.pretrain_log_interval
  # use_state_filter = config.use_state_filter

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

  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)

  logger = Logger(log_dir)

  env = cb_env_factory(**env_kwargs)  # type: ParallelEnv
  eval_env = cb_env_factory(**env_kwargs)  # type: ParallelEnv

  # Seed envs
  env.reset(seed=seed)
  eval_env.reset(seed=seed + 10)

  # ----- create agents
  dict_agents = {}  # type: Dict[Any, MA_OGAIL]
  for agent_idx in range(env.num_agents):
    a_name = env.agents[agent_idx]
    dict_agents[a_name] = make_agent(config, env, agent_idx, use_option)

  # ----- Load expert data
  n_labeled = int(n_traj * config.supervision)
  dict_expert_trajs, dict_expert_labels, cnt_label, _ = (
      load_multiagent_data_w_labels(env.agents, dict_agents, demo_path, n_traj,
                                    n_labeled, seed))

  # ----- add others' actions to each sample (for MA-GAIL critic)
  epi_vec_actions = []
  n_traj = len(dict_expert_trajs[env.agents[0]]["states"])
  for i_e in range(n_traj):
    list_joint_actions = []
    for i_t in range(dict_expert_trajs[env.agents[0]]["lengths"][i_e]):
      joint_a = []
      for a_name in env.agents:
        joint_a.append(dict_expert_trajs[a_name]["actions"][i_e][i_t])

      list_joint_actions.append(joint_a)
    vec_actions = list(zip(*list_joint_actions))
    vec_actions = [np.vstack(item) for item in vec_actions]
    epi_vec_actions.append(vec_actions)

  for i_a in range(len(env.agents)):
    a_name = env.agents[i_a]
    expert_trajs = dict_expert_trajs[a_name]
    agent = dict_agents[a_name]
    agent.action_dim
    agent.discrete_act
    agent.OTHERS_ACTION_SPLIT_SIZE
    list_others_actions = []
    for i_e in range(n_traj):
      vec_acts = epi_vec_actions[i_e]

      others_actions = np.hstack(vec_acts[:i_a] + vec_acts[i_a + 1:])
      list_others_actions.append(others_actions)

    expert_trajs["others_actions"] = list_others_actions

  # wandb.run.summary["expert_avg"] = expert_avg
  # wandb.run.summary["expert_std"] = expert_std

  if not os.path.exists(save_dir):
    os.mkdir(save_dir)

  best_model_save_name = os.path.join(
      save_dir, f"{env_name}_n{n_traj}_l{cnt_label}_best")

  explore_step = 0
  best_reward = -float("inf")
  cnt_evals = 0
  for i in count():
    if explore_step >= max_exp_step:
      wandb.finish()
      return

    # ----- infer mental state and prepare data
    if use_option:
      for a_name in env.agents:
        agent = dict_agents[a_name]
        expert_trajs = dict_expert_trajs[a_name]
        mental_states = infer_mental_states_all_demo(agent, expert_trajs,
                                                     dict_expert_labels[a_name])
        expert_trajs["latents"] = mental_states
        # create "prev_latents"
        init_lat = np.array(agent.PREV_LATENT).reshape(-1)
        list_prev_lat = []
        for i_e in range(len(expert_trajs["states"])):
          tmp_lat = np.array(mental_states[i_e][:-1]).reshape(-1, len(init_lat))
          expert_prev_lat = np.vstack([init_lat, tmp_lat])
          list_prev_lat.append(expert_prev_lat)
        expert_trajs["prev_latents"] = list_prev_lat

    # ----- collect data
    explore_step_cur = 0
    all_episode_rewards = {a_name: [] for a_name in env.agents}
    dict_online_trajs = {a_name: defaultdict(list) for a_name in env.agents}
    list_wins = []

    for n_epi in count():
      episode_rewards = {a_name: 0 for a_name in env.agents}
      episode_tuples = {a_name: [] for a_name in env.agents}
      env_done = False

      joint_obs, infos = env.reset()
      joint_prev_lat = {}
      joint_prev_aux = {}
      for a_name in env.agents:
        agent = dict_agents[a_name]
        prev_lat, prev_aux = agent.PREV_LATENT, agent.PREV_AUX
        joint_prev_lat[a_name] = prev_lat
        joint_prev_aux[a_name] = prev_aux

      for episode_step in count():
        joint_actions = {}
        joint_latents = {}
        list_actions = []
        for a_name in env.agents:
          if "avail_actions" in infos[a_name]:
            available_actions = np.array(infos[a_name]["avail_actions"])
          else:
            available_actions = None

          option, action = dict_agents[a_name].choose_action(
              joint_obs[a_name],
              joint_prev_lat[a_name],
              joint_prev_aux[a_name],
              sample=True,
              avail_actions=available_actions)
          joint_latents[a_name] = option
          joint_actions[a_name] = action
          list_actions.append(action)

        joint_next_obs, rewards, dones, truncates, infos = env.step(
            joint_actions)
        joint_aux = {}
        for i_agent in range(len(env.agents)):
          a_name = env.agents[i_agent]
          if config.use_auxiliary_obs:
            joint_aux[a_name] = env.get_auxiliary_obs(a_name)
          else:
            joint_aux[a_name] = dict_agents[a_name].PREV_AUX

          others_actions = np.hstack(list_actions[:i_agent] +
                                     list_actions[i_agent + 1:])
          episode_rewards[a_name] += rewards[a_name]
          episode_tuples[a_name].append(
              (joint_prev_lat[a_name], joint_prev_aux[a_name],
               joint_obs[a_name], joint_latents[a_name], joint_actions[a_name],
               joint_aux[a_name], joint_next_obs[a_name], rewards[a_name],
               dones[a_name], others_actions))

          if dones[a_name] or truncates[a_name]:
            env_done = True

        explore_step_cur += 1
        if env_done:
          break

        joint_obs = joint_next_obs
        joint_prev_lat = joint_latents
        joint_prev_aux = joint_aux

      EPI_ITEM_KEYS = [
          'prev_latents', 'prev_auxs', 'states', 'latents', 'actions', 'auxs',
          'next_states', 'rewards', 'dones', 'others_actions'
      ]
      for a_name in env.agents:
        all_episode_rewards[a_name].append(episode_rewards[a_name])
        vec_epi_items = zip(*episode_tuples[a_name])
        for idx, epi_item in enumerate(vec_epi_items):
          dict_online_trajs[a_name][EPI_ITEM_KEYS[idx]].append(epi_item)

        dict_online_trajs[a_name]["lengths"].append(len(episode_tuples[a_name]))
      if 'won' in infos[env.agents[0]]:
        if infos[env.agents[0]]['won']:
          list_wins.append(1)
        else:
          list_wins.append(0)

      if explore_step_cur >= n_sample:
        break

    ret_sum = np.zeros_like(all_episode_rewards[env.agents[0]])
    for a_name in env.agents:
      ret_sum = ret_sum + np.array(all_episode_rewards[a_name])
      logger.log_train(f"returns/{a_name}",
                       np.mean(all_episode_rewards[a_name]), explore_step)
    avg_return = np.mean(ret_sum)
    avg_epi_step = explore_step_cur / (n_epi + 1)
    logger.log_train("episode_reward", avg_return, explore_step)
    logger.log_train("episode_step", avg_epi_step, explore_step)
    logger.log_train("win_rate", np.mean(list_wins), explore_step)
    print(f"{explore_step}: episode_reward={avg_return}, "
          f"episode_step={avg_epi_step} ; {env_name}_{run_name}")

    # ----- replace rewards with discriminator rewards
    for a_name in env.agents:
      online_trajs = dict_online_trajs[a_name]
      for i_e in range(len(online_trajs["states"])):
        online_trajs["rewards"][i_e] = dict_agents[a_name].gail_reward(
            online_trajs["states"][i_e], online_trajs["prev_latents"][i_e],
            online_trajs["prev_auxs"][i_e], online_trajs["actions"][i_e],
            online_trajs["latents"][i_e])

    # ----- update models
    for a_name in env.agents:
      dict_agents[a_name].gail_update(dict_online_trajs[a_name],
                                      dict_expert_trajs[a_name],
                                      n_step=config.n_update_rounds)
      dict_agents[a_name].ppo_update(dict_online_trajs[a_name],
                                     n_step=config.n_update_rounds)

    explore_step += explore_step_cur

    # ----- evaluate models
    cnt_evals += explore_step_cur
    if cnt_evals >= eval_interval:
      cnt_evals = 0

      dict_eval_returns, eval_timesteps, wins = evaluate(
          dict_agents,
          eval_env,
          config.use_auxiliary_obs,
          num_episodes=num_episodes)
      ret_sum = np.zeros_like(dict_eval_returns[env.agents[0]])
      for a_name in env.agents:
        ret_sum = ret_sum + np.array(dict_eval_returns[a_name])
        logger.log_eval(f"returns/{a_name}", np.mean(dict_eval_returns[a_name]),
                        explore_step)

      mean_ret_sum = np.mean(ret_sum)
      mean_eval_step = np.mean(eval_timesteps)
      logger.log_eval('episode_reward', mean_ret_sum, explore_step)
      logger.log_eval('episode_step', mean_eval_step, explore_step)
      logger.log_eval('win_rate', np.mean(wins), explore_step)
      print(f"{explore_step}[Eval]: episode_reward={mean_ret_sum}, "
            f"episode_step={mean_eval_step} ; {env_name}_{run_name}")

      if mean_ret_sum >= best_reward:
        best_reward = mean_ret_sum
        wandb.run.summary["best_returns"] = best_reward
        for a_name in env.agents:
          dict_agents[a_name].save(best_model_save_name + f"_{a_name}")

    logger.flush()

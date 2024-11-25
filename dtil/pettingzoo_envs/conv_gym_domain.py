import os
import pickle
import functools
from collections import defaultdict
import gymnasium
from gymnasium.spaces import Discrete, Box
from gymnasium.utils import seeding
import numpy as np
from pettingzoo import ParallelEnv
import gym
import gym_custom
from gym_custom.envs.multiple_goals_2d import MGExpert
from stable_baselines3.common.monitor import Monitor


class ConvGymDomain(ParallelEnv):

  def __init__(self, env_name, render_mode="human") -> None:
    self.render_mode = render_mode
    self.possible_agents = [0]

    env_kwargs = {}
    self.gym_env = gym.make(env_name, **env_kwargs)
    self.gym_env = Monitor(self.gym_env, "gym")

  @functools.lru_cache(maxsize=None)
  def observation_space(self, agent):
    return self.gym_env.observation_space

  @functools.lru_cache(maxsize=None)
  def action_space(self, agent):
    return self.gym_env.action_space

  def render(self):
    return self.gym_env.render(self.render_mode)

  def close(self):
    return self.gym_env.close()

  def _seed(self, seed=None):
    self.np_random, seed = seeding.np_random(seed)

  def reset(self, seed=None, options=None):
    if seed is not None:
      self._seed(seed)
    self.agents = self.possible_agents

    obs = self.gym_env.reset()

    return {0: obs}, {0: {}}

  def step(self, actions):
    action = actions[0]
    obs, reward, done, info = self.gym_env.step(action)
    trunc = info.get('TimeLimit.truncated', False)

    return {0: obs}, {0: reward}, {0: done}, {0: trunc}, {0: info}


def generate_data(save_dir, env_name, n_traj, render=False, render_delay=10):
  expert_trajs = defaultdict(list)

  env = ConvGymDomain(env_name)
  if hasattr(env.gym_env, "set_render_delay"):
    env.gym_env.set_render_delay(render_delay)

  agents = {0: MGExpert(env.gym_env, 0.3)}

  for _ in range(n_traj):
    obs, infos = env.reset()
    prev_latents = {aname: agents[aname].PREV_LATENT for aname in env.agents}
    episode_reward = {aname: 0 for aname in env.agents}

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

      prev_latents = latents
      obs = next_obs

      if render:
        env.render()

      if all(truncs.values()) or all(dones.values()):
        break

    if render:
      print(episode_reward, len(samples))

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

  if save_dir is not None:
    file_path = os.path.join(save_dir, f"{env_name}_{n_traj}.pkl")
    with open(file_path, 'wb') as f:
      pickle.dump(expert_trajs, f)

  return expert_trajs


if __name__ == "__main__":
  cur_dir = os.path.dirname(__file__)

  #   traj = generate_data(cur_dir, "MultiGoals2D_2-v0", 500, False, 100)
  traj = generate_data(None, "MultiGoals2D_2-v0", 1, True, 100)

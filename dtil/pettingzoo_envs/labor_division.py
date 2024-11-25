import os
import random
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


class LaborDivision(ParallelEnv):
  metadata = {"render_modes": ["human"], "name": "multi_professionals"}

  def __init__(self, targets, n_agents, render_mode=None):
    """
        The init method takes in environment arguments and should define 
        the following attributes:
        - possible_agents
        - render_mode

        obs: 
        (position, my_action, progressing,
        {observed, teammate_rel_position, teammate_action})
    """

    self.render_mode = render_mode
    self.possible_agents = [idx for idx in range(n_agents)]

    self.agent_name_mapping = dict(
        zip(self.possible_agents, list(range(len(self.possible_agents)))))

    WORLD_HALF_SIZE_X = 5
    WORLD_HALF_SIZE_Y = 5

    self.vis_rad = 2
    self.world_low = np.array([-WORLD_HALF_SIZE_X, -WORLD_HALF_SIZE_Y])
    self.world_high = np.array([WORLD_HALF_SIZE_X, WORLD_HALF_SIZE_Y])

    # define task world
    self.max_step = 50
    self.np_targets = np.array(targets)
    self.dict_agent_pos = {}
    #  -  target status: 0 means ready. positive values mean progress
    #                    negative values mean remaining time to get ready
    #                    each column corresponds to each agent.
    self.restock_time = 15
    self.max_progress = 5
    self.target_status = np.zeros((len(self.np_targets), n_agents))
    self.tolerance = 0.5
    self.noise_amp = 0.1

    # define visualization parameters
    self.unit_scr_sz = 30
    self.delay = 10

    curdir = os.path.dirname(__file__)
    img_location = cv2.imread(curdir + '/images/conveyor.png',
                              cv2.IMREAD_UNCHANGED)[:, :, :3]
    img_man = cv2.imread(curdir + '/images/man.png',
                         cv2.IMREAD_UNCHANGED)[:, :, :3]
    img_woman = cv2.imread(curdir + '/images/woman.png',
                           cv2.IMREAD_UNCHANGED)[:, :, :3]

    self.img_location = cv2.resize(img_location, (50, 50))
    self.img_agents = [
        cv2.resize(img_man, (30, 40)),
        cv2.resize(img_woman, (30, 40))
    ]

  @functools.lru_cache(maxsize=None)
  def observation_space(self, agent):
    # gymnasium spaces are defined and documented here:
    #    https://gymnasium.farama.org/api/spaces/
    n_agents = len(self.possible_agents)
    obs_low = np.array([*self.world_low, -1, -1, 0] +
                       [0, -self.vis_rad, -self.vis_rad, -1, -1] *
                       (n_agents - 1))
    obs_high = np.array([*self.world_high, 1, 1, 1] +
                        [1, self.vis_rad, self.vis_rad, 1, 1] * (n_agents - 1))

    return Box(low=obs_low, high=obs_high, dtype=np.float32)

  @functools.lru_cache(maxsize=None)
  def action_space(self, agent):
    return Box(low=-1, high=1, shape=(2, ), dtype=np.float32)

  def env_pt_2_scr_pt(self, env_pt):
    pt = env_pt - self.world_low
    pt = pt * self.unit_scr_sz
    return pt.astype(np.int64)

  def set_render_delay(self, delay):
    self.delay = delay

  def _draw_background(self, canvas):
    canvas_new = np.copy(canvas)
    for idx, target in enumerate(self.np_targets):
      target_pt = self.env_pt_2_scr_pt(target)

      x_p = int(target_pt[0] - self.img_location.shape[1] / 2)
      y_p = int(target_pt[1] - self.img_location.shape[0] / 2)
      canvas_new[y_p:y_p + self.img_location.shape[0],
                 x_p:x_p + self.img_location.shape[1]] = self.img_location

      # color = (255, 0, 0)
      # canvas_new = cv2.circle(canvas_new, target_pt,
      #                         int(self.tolerance * self.unit_scr_sz), color, -1)

    return canvas_new

  def get_canvas(self):
    cnvs_sz = self.unit_scr_sz * (self.world_high - self.world_low)
    canvas = np.ones((*cnvs_sz, 3), dtype=np.uint8) * 255
    canvas = self._draw_background(canvas)
    colors = [(0, 0, 255), (255, 0, 0), (255, 255, 0)]
    for aidx, agent in enumerate(self.possible_agents):
      pos = self.dict_agent_pos[agent]
      pos_pt = self.env_pt_2_scr_pt(pos)

      x_p = int(pos_pt[0] - self.img_agents[aidx].shape[1] / 2)
      y_p = int(pos_pt[1] - self.img_agents[aidx].shape[0] / 2)
      canvas[y_p:y_p + self.img_agents[aidx].shape[0],
             x_p:x_p + self.img_agents[aidx].shape[1]] = self.img_agents[aidx]

      color = colors[aidx]
      # canvas = cv2.circle(canvas, pos_pt, 10, color, -1)
      canvas = cv2.circle(canvas, pos_pt, int(self.vis_rad * self.unit_scr_sz),
                          color, 1)

    return canvas

  def render(self):
    if self.render_mode is None:
      gymnasium.logger.warn(
          "You are calling render method without specifying any render mode.")
      return
    elif self.render_mode == "human":
      cv2.imshow("LaborDivision", self.get_canvas())
      cv2.waitKey(self.delay)
      return
    pass

  def close(self):
    cv2.destroyAllWindows()
    pass

  def _seed(self, seed=None):
    # self.np_random, seed = seeding.np_random(seed)
    np.random.seed(seed)
    random.seed(seed)

  def _closest_target(self, agent):
    tidx = -1
    min_dist = 99999
    pos = self.dict_agent_pos[agent]
    for idx, target in enumerate(self.np_targets):
      dist = np.linalg.norm(target - pos)
      if min_dist > dist:
        min_dist = dist
        tidx = idx

    return tidx

  def _compute_obs(self, actions):
    dict_obs = {}
    for agent_me in self.possible_agents:
      my_pos = self.dict_agent_pos[agent_me]
      obs = [my_pos, actions[agent_me]]

      # progressing?
      progressing, _ = self._get_progressing_state(agent_me)
      obs.append(progressing)

      # relative pos of other agents
      for agent_fr in self.possible_agents:
        if agent_me == agent_fr:
          continue
        # relative position. if out of visible range, set it as not-observed
        rel_pos = self.dict_agent_pos[agent_fr] - my_pos
        fr_state = [1, *rel_pos, *actions[agent_fr]]
        if np.linalg.norm(rel_pos) > self.vis_rad:
          fr_state = [0, 0, 0, 0, 0]

        obs.append(fr_state)

      dict_obs[agent_me] = np.hstack(obs)

    return dict_obs

  def _get_progressing_state(self, agent):
    'return progressing state and the nearest target index.'
    tidx = self._closest_target(agent)
    tar_pos = self.np_targets[tidx]
    dist2tar = np.linalg.norm(tar_pos - self.dict_agent_pos[agent])
    tar_status = self.target_status[tidx, self.agent_name_mapping[agent]]

    # to progress, an agent has to be close to the target,
    # the target has a job to progress, and
    # there is no other agent around the target,
    progressing = 0
    if (dist2tar <= self.tolerance and tar_status >= 0
        and tar_status < self.max_progress):
      if self._get_dist_other2target(agent, tar_pos) > self.tolerance:
        progressing = 1

    return progressing, tidx

  def _get_dist_other2target(self, agent, tar_pos):
    dist_fr2tar = 99999
    for agent_fr in self.possible_agents:
      if agent == agent_fr:
        continue

      # other friend closest to the target
      tmp_dist = np.linalg.norm(tar_pos - self.dict_agent_pos[agent_fr])
      if tmp_dist < dist_fr2tar:
        dist_fr2tar = tmp_dist

    return dist_fr2tar

  def reset(self, seed=None, options=None):
    """
        Reset needs to initialize the `agents` attribute and must set up the
        environment so that render(), and step() can be called without issues.
        Here it initializes the `num_moves` variable which counts the number of
        hands that are played.
        Returns the observations for each agent
        """
    if seed is not None:
      self._seed(seed)

    self.agents = self.possible_agents[:]
    n_agents = len(self.agents)
    self.cur_step = 0

    margin = 0.5
    if options is not None:
      self.dict_agent_pos = options
    else:
      self.dict_agent_pos = {
          aname:
          np.random.uniform(self.world_low + margin, self.world_high - margin)
          for aname in range(n_agents)
      }
    self.target_status = np.zeros((len(self.np_targets), n_agents))

    actions = {agent: np.zeros(2) for agent in self.agents}
    observations = self._compute_obs(actions)
    infos = {agent: {} for agent in self.agents}

    return observations, infos

  def step(self, actions):
    """
        step(action) takes in an action for each agent and should return the
        - observations
        - rewards
        - terminations
        - truncations
        - infos
        dicts where each dict looks like {agent_1: item_1, agent_2: item_2}
        """

    self.cur_step += 1
    trunc = self.cur_step >= self.max_step

    next_agent_pos = {}
    for agent in self.possible_agents:
      action_clipped = np.clip(actions[agent], -1, 1)
      noise = (2 * np.random.rand(2) - 1) * self.noise_amp
      pos = self.dict_agent_pos[agent] + np.array(action_clipped) + noise
      pos[0] = min(
          self.observation_space(agent).high[0],
          max(self.observation_space(agent).low[0], pos[0]))
      pos[1] = min(
          self.observation_space(agent).high[1],
          max(self.observation_space(agent).low[1], pos[1]))
      next_agent_pos[agent] = pos

    PENALTY = -0.0
    POINT = 1

    rewards = {agent: PENALTY for agent in self.agents}
    for agent in self.possible_agents:
      progressing, tidx = self._get_progressing_state(agent)
      if progressing:
        rewards[agent] += POINT
        aidx = self.agent_name_mapping[agent]
        if self.target_status[tidx, aidx] == self.max_progress - 1:
          self.target_status[tidx, aidx] = -(self.restock_time + 1)
        else:
          self.target_status[tidx, aidx] += 1

    # reduce waiting time by 1
    self.target_status[self.target_status < 0] += 1

    self.dict_agent_pos = next_agent_pos

    observations = self._compute_obs(actions)
    infos = {agent: {} for agent in self.agents}

    dones = {agent: False for agent in self.agents}
    truncs = {agent: trunc for agent in self.agents}

    return observations, rewards, dones, truncs, infos


class DyadLaborDivision(LaborDivision):

  def __init__(self, targets, render_mode=None):
    super().__init__(targets, n_agents=2, render_mode=render_mode)


class TwoTargetDyadLaborDivision(DyadLaborDivision):

  def __init__(self, render_mode=None):
    super().__init__([(-4, 0), (4, 0)], render_mode)


class ThreeTargetDyadLaborDivision(DyadLaborDivision):

  def __init__(self, render_mode=None):
    super().__init__([(-4, -3), (4, -3), (0, 4)], render_mode)


class LDExpert:

  def __init__(self, env, tolerance, name) -> None:
    self.PREV_LATENT = None
    self.PREV_AUX = None  # for compatibility with MAHIL agent
    self.env = env
    self.n_agents = len(env.possible_agents)
    self.np_targets = env.np_targets
    self.name = name

  def conv_obs(self, obs):
    pos = obs[0:2]
    act = obs[2:4]
    progressing = obs[4]

    # let's think about only dyad setting
    observed = obs[5]
    rel_pos = obs[6:8]
    act_fr = obs[8:10]
    return pos, act, progressing, observed, rel_pos, act_fr

  def choose_mental_state(self, obs, prev_latent, aux=None, sample=False):
    # NOTE:
    # near target / progressing
    #   --> maintain current target regardless of other agent
    # near target / not progressing
    #   --> need to change target. predict where the other agent goes.
    #     --> if it comes to current one / stays --> i go the other one
    #     --> if it goes the other one
    #       -->if there is remaining target --> go to the remaining one
    #       --> no remaining target --> either stay here or go to other
    #         --> if i am closer to the target --> change to that one
    #         --> otherwise --> stay put
    # not near target --> predict where the other agent goes
    #   --> if friend is near target
    #     --> going to the target / staying --> i go to the other target
    #     --> heading to other target --> maintain my target
    #   --> friend not near target
    #     --> friend go to other target --> maintain my target
    #     --> friend go to my target --> i change target only if i am further.

    if prev_latent == self.PREV_LATENT:
      return np.random.choice(range(len(self.np_targets)))

    pos, act, progressing, observed, rel_pos, act_fr = self.conv_obs(obs)

    # find the closest target
    clst_tidx = -1
    min_dist = 999999
    for tidx, target in enumerate(self.np_targets):
      dist = np.linalg.norm(target - pos)
      if min_dist > dist:
        min_dist = dist
        clst_tidx = tidx

    not_clst_tidx = [
        tidx for tidx in range(len(self.np_targets)) if tidx != clst_tidx
    ]
    not_prev_tidx = [
        tidx for tidx in range(len(self.np_targets)) if tidx != prev_latent
    ]

    prev_neq_clst = prev_latent != clst_tidx
    sample_from_not_clst = prev_latent if prev_neq_clst else np.random.choice(
        not_clst_tidx)
    sample_from_not_prev = np.random.choice(not_prev_tidx)

    # no nearby agent
    if observed == 0:
      # near a target but not progressing
      if min_dist <= self.env.tolerance and progressing == 0:
        return sample_from_not_clst
      else:
        return prev_latent
    # agents exist nearby
    else:
      # progressing  --> maintain current target regardless of friend
      if progressing == 1:
        return prev_latent
      # not progressing --> may need changing target. predict where friend goes.
      else:
        # predict which target the friend is going
        pos_fr = pos + rel_pos  # friend position
        prev_pos_fr = pos_fr - act_fr
        fr_staying = np.linalg.norm(act_fr) < 0.2
        max_inner = -9999999
        fr_tidx = -1
        for tidx, target in enumerate(self.np_targets):
          direction = target - prev_pos_fr
          len_dir = np.linalg.norm(direction)
          if len_dir != 0:
            direction /= len_dir
          inner = np.dot(direction, act_fr)
          if inner > max_inner:
            max_inner = inner
            fr_tidx = tidx

        # near target.
        if min_dist <= self.env.tolerance:
          # if friend comes to current one / stays put --> i go the other one
          if fr_tidx == clst_tidx or fr_staying:
            return sample_from_not_clst
          # if friend goes to the other one
          else:
            # if there is remaining target --> go to the remaining one
            if len(self.np_targets) > 2:
              remaining_tidx = []
              for tidx in not_clst_tidx:
                if fr_tidx != tidx:
                  remaining_tidx.append(tidx)
              return (prev_latent if prev_latent in remaining_tidx else
                      np.random.choice(remaining_tidx))
            # no remaining target --> either stay here or go to other
            else:
              # if i am closer to the current latent target --> go to that one
              if (np.linalg.norm(self.np_targets[prev_latent] - pos)
                  < np.linalg.norm(self.np_targets[prev_latent] - pos_fr)):
                return prev_latent
              else:
                return len(self.np_targets) - 1 - prev_latent
        # not near target
        else:
          # if friend is already near my target
          if (np.linalg.norm(self.np_targets[prev_latent] - pos_fr)
              <= self.env.tolerance):
            IMMEDIATE_CHANGE = False
            if IMMEDIATE_CHANGE:
              # friend is going to my target / staying there --> i go to the other one
              if fr_tidx == prev_latent or fr_staying:
                return sample_from_not_prev
              # friend heading to other target --> maintain my target
              else:
                return prev_latent
            else:  # maintain current latent (wait for friend to finish first)
              return prev_latent
          # friend is not near my target
          else:
            # friend goes to my target
            if fr_tidx == prev_latent:
              # if friend is closer to the target
              if (np.linalg.norm(self.np_targets[prev_latent] - pos_fr)
                  < np.linalg.norm(self.np_targets[prev_latent] - pos)):
                return sample_from_not_prev
              else:
                return prev_latent
            # friend goes to the other target
            else:
              return prev_latent

  def choose_policy_action(self, obs, latent, sample=False):

    pos, act, progressing, observed, rel_pos, act_fr = self.conv_obs(obs)

    pos_fr = pos + rel_pos
    # if friend is near the target while im not --> wait with some distance
    if observed == 1 and (np.linalg.norm(self.np_targets[latent] - pos_fr)
                          <= self.env.tolerance):
      target = self.np_targets[latent]
      vec_dir = target - pos
      len_dir = np.linalg.norm(vec_dir)

      if len_dir != 0:
        vec_dir /= len_dir
      else:
        d_ori2tar = np.linalg.norm(target)
        assert d_ori2tar > 0
        vec_dir = target / d_ori2tar

      len_move = np.clip(len_dir - (self.env.tolerance + 0.5), -0.7, 0.7)
      vec_dir *= len_move

      return vec_dir

    else:
      target = self.np_targets[latent]
      vec_dir = target - pos
      len_dir = np.linalg.norm(vec_dir)
      if len_dir >= 1:
        vec_dir /= len_dir

      return vec_dir

  def choose_action(self, obs, prev_latent, sample=False, **kwargs):
    option = self.choose_mental_state(obs, prev_latent, sample=sample)
    action = self.choose_policy_action(obs, option, sample)
    return option, action


class LDExpert_V2(LDExpert):

  def choose_mental_state(self, obs, prev_latent, aux=None, sample=False):
    if prev_latent == self.PREV_LATENT:
      return np.random.choice(range(len(self.np_targets)))

    pos, act, progressing, observed, rel_pos, act_fr = self.conv_obs(obs)

    # find the closest target
    clst_tidx = -1
    min_dist = 999999
    for tidx, target in enumerate(self.np_targets):
      dist = np.linalg.norm(target - pos)
      if min_dist > dist:
        min_dist = dist
        clst_tidx = tidx

    not_clst_tidx = [
        tidx for tidx in range(len(self.np_targets)) if tidx != clst_tidx
    ]

    prev_neq_clst = prev_latent != clst_tidx
    sample_from_not_clst = prev_latent if prev_neq_clst else np.random.choice(
        not_clst_tidx)

    # no nearby agent
    if observed == 0:
      # near a target but not progressing
      if min_dist <= self.env.tolerance and progressing == 0:
        return sample_from_not_clst
      else:
        return prev_latent
    # agents exist nearby
    else:
      # progressing  --> maintain current target regardless of friend
      if progressing == 1:
        return prev_latent
      # not progressing --> may need changing target.
      #                     no prediction regarding other agent action
      else:
        # all random sample
        return np.random.choice(range(len(self.np_targets)))


def generate_data(save_dir,
                  expert_class: Type[LDExpert],
                  env_name,
                  n_traj,
                  render=False,
                  render_delay=10):
  expert_trajs = defaultdict(list)
  if env_name == "LaborDivision2":
    env = TwoTargetDyadLaborDivision(render_mode="human")
  elif env_name == "LaborDivision3":
    env = ThreeTargetDyadLaborDivision(render_mode="human")
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

  # traj = generate_data(cur_dir, LDExpert, "LaborDivision2", 100, False, 300)
  # traj = generate_data(None, LDExpert, "LaborDivision2", 10, False, 100)
  # traj = generate_data(None, LDExpert_V2, "LaborDivision2", 100, True, 100)
  traj = generate_data(None, LDExpert_V2, "LaborDivision3", 100, True, 100)

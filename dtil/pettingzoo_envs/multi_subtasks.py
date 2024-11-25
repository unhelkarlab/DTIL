import os
import random
import math
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
from tqdm import tqdm


class MultiSubTasks(ParallelEnv):
  metadata = {"render_modes": ["human"], "name": "multi_professionals"}

  def __init__(self, n_targets, n_agents, render_mode=None):
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
    self.n_targets = n_targets
    self.dict_agent_pos = {}
    #  -  target status: 0 means ready. positive values mean progress
    #                    negative values mean remaining time to get ready
    #                    each column corresponds to each agent.
    self.restock_time = 15
    self.max_progress = 5
    self.target_status = np.zeros((n_targets, n_agents))
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
    self.n_targets
    n_agents = len(self.possible_agents)
    obs_low = np.array([*self.world_low, 0] +
                       [0, -self.vis_rad, -self.vis_rad] * (n_agents - 1) +
                       [*self.world_low] * self.n_targets)
    obs_high = np.array([*self.world_high, 1] +
                        [1, self.vis_rad, self.vis_rad] * (n_agents - 1) +
                        [*self.world_high] * self.n_targets)

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

      x_m = int(target_pt[0] - self.img_location.shape[1] / 2)
      y_m = int(target_pt[1] - self.img_location.shape[0] / 2)

      x_M = x_m + self.img_location.shape[1]
      y_M = y_m + self.img_location.shape[0]

      x_m = max(0, x_m)
      y_m = max(0, y_m)
      x_M = min(canvas_new.shape[1], x_M)
      y_M = min(canvas_new.shape[0], y_M)

      x_c = self.img_location.shape[1] // 2
      y_c = self.img_location.shape[0] // 2

      x_l = x_c - (target_pt[0] - x_m)
      x_h = x_c + (x_M - target_pt[0])
      y_l = y_c - (target_pt[1] - y_m)
      y_h = y_c + (y_M - target_pt[1])

      canvas_new[y_m:y_M, x_m:x_M] = self.img_location[y_l:y_h, x_l:x_h]

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

      x_m = int(pos_pt[0] - self.img_agents[aidx].shape[1] / 2)
      y_m = int(pos_pt[1] - self.img_agents[aidx].shape[0] / 2)

      x_M = x_m + self.img_agents[aidx].shape[1]
      y_M = y_m + self.img_agents[aidx].shape[0]

      x_m = max(0, x_m)
      y_m = max(0, y_m)
      x_M = min(canvas.shape[1], x_M)
      y_M = min(canvas.shape[0], y_M)

      x_c = self.img_agents[aidx].shape[1] // 2
      y_c = self.img_agents[aidx].shape[0] // 2

      x_l = x_c - (pos_pt[0] - x_m)
      x_h = x_c + (x_M - pos_pt[0])
      y_l = y_c - (pos_pt[1] - y_m)
      y_h = y_c + (y_M - pos_pt[1])

      canvas[y_m:y_M, x_m:x_M] = self.img_agents[aidx][y_l:y_h, x_l:x_h]

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

      # sub-task locations
      obs.append(self.np_targets.reshape(-1))

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

  @staticmethod
  def generate_targets(num_targets,
                       radius,
                       min_distance,
                       world_low,
                       world_high,
                       center_clear_radius=0):

    def distance(x1, y1, x2, y2):
      return math.sqrt((x1 - x2)**2 + (y1 - y2)**2)

    targets = []

    diameter = 2 * radius
    cen_x, cen_y = (world_low + world_high) / 2

    while len(targets) < num_targets:
      # Randomly place a new target within the bounds of the space (considering the radius)
      x = random.uniform(world_low[0] + radius, world_high[0] - radius)
      y = random.uniform(world_low[1] + radius, world_high[1] - radius)
      if distance(x, y, cen_x, cen_y) < center_clear_radius:
        continue

      # Check if the new target overlaps with any existing targets
      if all(
          distance(x, y, tx, ty) >= (diameter + min_distance)
          for tx, ty in targets):
        targets.append((x, y))

    return targets

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

    # sample target positions
    targets = MultiSubTasks.generate_targets(self.n_targets,
                                             self.tolerance,
                                             4,
                                             self.world_low,
                                             self.world_high,
                                             center_clear_radius=3)
    self.np_targets = np.array(targets)

    self.target_status = np.zeros((self.n_targets, n_agents))

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


class MultiSubTasksDyadTwoTargets(MultiSubTasks):

  def __init__(self, render_mode=None):
    super().__init__(n_targets=2, n_agents=2, render_mode=render_mode)


class MultiSubTasksDyadThreeTargets(MultiSubTasks):

  def __init__(self, render_mode=None):
    super().__init__(n_targets=3, n_agents=2, render_mode=render_mode)


class MultiSubTasksExpert:

  def __init__(self, n_agents, n_targets, tolerance, name) -> None:
    self.PREV_LATENT = None
    self.PREV_AUX = None  # for compatibility with MAHIL agent
    self.n_agents = n_agents
    self.n_targets = n_targets
    self.tolerance = tolerance
    self.name = name

  def conv_obs(self, obs):
    pos = obs[0:2]
    progressing = obs[2]

    # let's think about only dyad setting
    observed = obs[3]
    rel_pos = obs[4:6]

    np_targets = obs[6:].reshape(-1, 2)

    return pos, progressing, observed, rel_pos, np_targets

  def choose_mental_state(self, obs, prev_latent, aux=None, sample=False):
    if prev_latent == self.PREV_LATENT:
      return np.random.choice(range(self.n_targets))

    pos, progressing, observed, rel_pos, np_targets = self.conv_obs(obs)

    # find the closest target
    clst_tidx = -1
    min_dist = 999999
    for tidx, target in enumerate(np_targets):
      dist = np.linalg.norm(target - pos)
      if min_dist > dist:
        min_dist = dist
        clst_tidx = tidx

    not_clst_tidx = [
        tidx for tidx in range(len(np_targets)) if tidx != clst_tidx
    ]

    prev_neq_clst = prev_latent != clst_tidx
    sample_from_not_clst = prev_latent if prev_neq_clst else np.random.choice(
        not_clst_tidx)

    # no nearby agent
    if observed == 0:
      # near a target but not progressing
      if min_dist <= self.tolerance and progressing == 0:
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
        return np.random.choice(range(len(np_targets)))

  def choose_policy_action(self, obs, latent, sample=False):

    pos, progressing, observed, rel_pos, np_targets = self.conv_obs(obs)

    pos_fr = pos + rel_pos
    # if friend is near the target while im not --> wait with some distance
    if observed == 1 and (np.linalg.norm(np_targets[latent] - pos_fr)
                          <= self.tolerance):
      target = np_targets[latent]
      vec_dir = target - pos
      len_dir = np.linalg.norm(vec_dir)

      if len_dir != 0:
        vec_dir /= len_dir
      else:
        d_ori2tar = np.linalg.norm(target)
        assert d_ori2tar > 0
        vec_dir = target / d_ori2tar

      len_move = np.clip(len_dir - (self.tolerance + 0.5), -0.7, 0.7)
      vec_dir *= len_move

      return vec_dir

    else:
      target = np_targets[latent]
      vec_dir = target - pos
      len_dir = np.linalg.norm(vec_dir)
      if len_dir >= 1:
        vec_dir /= len_dir

      return vec_dir

  def choose_action(self, obs, prev_latent, sample=False, **kwargs):
    option = self.choose_mental_state(obs, prev_latent, sample=sample)
    action = self.choose_policy_action(obs, option, sample)
    return option, action


def generate_data(save_dir,
                  expert_class: Type[MultiSubTasksExpert],
                  env_name,
                  n_traj,
                  render=False,
                  render_delay=10):
  expert_trajs = defaultdict(list)
  if env_name == "MultiSubTasks2":
    env = MultiSubTasksDyadTwoTargets(render_mode="human")
  elif env_name == "MultiSubTasks3":
    env = MultiSubTasksDyadThreeTargets(render_mode="human")
  else:
    raise NotImplementedError()

  env.reset()

  n_targets = env.n_targets
  n_agents = env.num_agents
  tolerance = env.tolerance
  env.set_render_delay(render_delay)
  agents = {
      aname: expert_class(n_agents, n_targets, tolerance, aname)
      for aname in env.possible_agents
  }

  list_total_reward = []
  for _ in tqdm(range(n_traj)):
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

  # traj = generate_data(None, MultiSubTasksExpert, "MultiSubTasks3", 100,
  #                      True, 10)
  traj = generate_data(cur_dir, MultiSubTasksExpert, "MultiSubTasks2", 100,
                       False, 10)

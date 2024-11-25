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
from pettingzoo import ParallelEnv
from tqdm import tqdm
from omegaconf import DictConfig
from gym_cooking.utils.astar import (manhattan_distance,
                                     get_gridworld_astar_distance)
from gym_cooking.envs.overcooked_environment import OvercookedEnvironment
from gym_cooking.utils.core import GridSquare


class Overcooked(ParallelEnv):
  metadata = {"render_modes": ["human"], "name": "overcooked"}

  def __init__(self,
               level='open-divider_salad',
               num_agents=2,
               max_num_timesteps=100,
               max_num_subtasks=14,
               render_mode=None):

    self.render_mode = render_mode

    arglist = DictConfig({})
    arglist['num_agents'] = num_agents
    arglist['level'] = level
    arglist['max_num_timesteps'] = max_num_timesteps
    arglist['max_num_subtasks'] = max_num_subtasks
    arglist['play'] = False

    self.env = OvercookedEnvironment(arglist)
    obs, info = self.env.reset()

    self.possible_agents = self.env.get_agent_names()
    self.agents = self.possible_agents[:]

    self.agent_name_mapping = dict(
        zip(self.possible_agents, list(range(len(self.possible_agents)))))

    self.max_step = max_num_timesteps

    observation_size = len(obs)
    self.observation_spaces = {}
    for name in self.agents:
      self.observation_spaces[name] = Box(low=0,
                                          high=1,
                                          shape=(observation_size, ),
                                          dtype=np.float32)

    self.possible_actions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
    self.action_mapping = {
        act: idx
        for idx, act in enumerate(self.possible_actions)
    }

    self.action_spaces = {}
    for name in self.agents:
      self.action_spaces[name] = Discrete(len(self.possible_actions))

  @functools.lru_cache(maxsize=None)
  def observation_space(self, agent):
    return self.observation_spaces[agent]

  @functools.lru_cache(maxsize=None)
  def action_space(self, agent):
    return self.action_spaces[agent]

  def close(self):
    self.env.close()

  def reset(self, seed=None, options=None):
    obs, info = self.env.reset()

    self.agents = self.possible_agents[:]
    self.cur_step = 0

    infos = {}
    obs_dict = {}
    for aname in self.agents:
      infos[aname] = {"env_new": info["env_new"]}
      obs_dict[aname] = obs

    return obs_dict, infos

  def step(self, actions):

    self.cur_step += 1

    action_dict = {}
    for aname in self.agents:
      action_dict[aname] = self.possible_actions[actions[aname]]

    obs, reward, done, info = self.env.step(action_dict)

    dict_obs = {}
    dict_rew = {}
    dones = {}
    truncs = {}
    infos = {}
    for aname in self.agents:
      dict_obs[aname] = obs
      dict_rew[aname] = reward
      infos[aname] = {"env_new": info["env_new"]}
      if done:
        dones[aname] = self.env.successful
        if not self.env.successful:
          truncs[aname] = True
      else:
        dones[aname] = False
        truncs[aname] = False

    return dict_obs, dict_rew, dones, truncs, infos


# ---------------------------------------------------------------------------


def find_path(world, my_agent, obj_list, others_pos=None):
  # astar search / check holding
  my_pos = my_agent.location

  def get_neighbors(pos):
    obj_neighbors = []
    for x, y in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
      x_c, y_c = (pos[0] + x, pos[1] + y)
      if x_c >= 0 and x_c < world.width and y_c >= 0 and y_c < world.height:
        nei_pos = (x_c, y_c)
        if others_pos is not None and nei_pos in others_pos:
          continue

        gs = world.loc_to_gridsquare[nei_pos]
        if not gs.collidable:
          obj_neighbors.append(nei_pos)

    return obj_neighbors

  dict_target_obj = {}
  list_goals = []
  for obj in obj_list:
    neighbors = get_neighbors(obj.location)
    for coord in neighbors:
      dict_target_obj[coord] = dict_target_obj.get(coord, []) + [obj]

    list_goals = list_goals + neighbors

  list_goals = list(set(list_goals))
  if my_pos in list_goals:
    path = [my_pos]
  else:
    path = get_gridworld_astar_distance(my_pos,
                                        list_goals,
                                        cb_get_neighbors=get_neighbors,
                                        hueristic_fn=manhattan_distance)

  targeted_objs = []
  if len(path) > 0:
    targeted_objs = dict_target_obj[path[-1]]

  return path, targeted_objs


def reachable(world, agent, obj_list, others_pos=None):
  return len(find_path(world, agent, obj_list, others_pos)[0]) > 0


def get_world_info(obs):
  item_fullnames = []
  item_names = []
  item_2_hold_agents = {}
  item_2_hold_gs = {}
  list_cutboards = []
  list_delivery = []
  list_emptycounters = []
  for i_a, agent in enumerate(obs.sim_agents):
    x, y = agent.location
    if agent.holding is not None:
      item_fullnames.append(agent.holding.full_name)
      item_name = agent.holding.name
      item_names.append(item_name)
      item_2_hold_agents[item_name] = item_2_hold_agents.get(item_name,
                                                             []) + [agent]

  for obj in obs.world.get_object_list():
    if obj.name == "Cutboard":
      list_cutboards.append(obj)
    elif obj.name == "Delivery":
      list_delivery.append(obj)
    elif obj.name == "Counter" and obj.holding is None:
      list_emptycounters.append(obj)

    if obj.name not in ["Counter", "Cutboard"] or obj.holding is None:
      continue

    item_fullnames.append(obj.holding.full_name)
    item_name = obj.holding.name
    item_names.append(item_name)
    item_2_hold_gs[item_name] = item_2_hold_gs.get(item_name, []) + [obj]

  for iname in item_names:
    if iname not in item_2_hold_agents:
      item_2_hold_agents[iname] = []
    if iname not in item_2_hold_gs:
      item_2_hold_gs[iname] = []

  dict_info = {
      'item_fullnames': item_fullnames,
      'item_2_hold_agents': item_2_hold_agents,
      'item_2_hold_gs': item_2_hold_gs,
      'list_cutboards': list_cutboards,
      'list_delivery': list_delivery,
      'list_emptycounters': list_emptycounters
  }

  return dict_info


def assign_subtasks(obs, list_agent_subtasks):
  all_subtasks = obs.all_subtasks
  dict_info = get_world_info(obs)
  item_fullnames = dict_info["item_fullnames"]
  item_2_hold_agents = dict_info["item_2_hold_agents"]
  item_2_hold_gs = dict_info["item_2_hold_gs"]
  list_cutboards = dict_info["list_cutboards"]
  list_delivery = dict_info["list_delivery"]

  possible_subtasks = []
  if 'FreshTomato' in item_fullnames:
    possible_subtasks.append(('Chop', 'Tomato'))

  if 'FreshLettuce' in item_fullnames:
    possible_subtasks.append(('Chop', 'Lettuce'))

  chopped_tomato = 'ChoppedTomato' in item_fullnames
  chopped_lettuce = 'ChoppedLettuce' in item_fullnames
  if chopped_tomato:
    possible_subtasks.append(('Merge', 'Tomato', 'Plate'))

  if chopped_lettuce:
    possible_subtasks.append(('Merge', 'Lettuce', 'Plate'))

  if chopped_tomato and chopped_lettuce:
    possible_subtasks.append(('Merge', 'Tomato', 'Lettuce'))

  if 'ChoppedLettuce-ChoppedTomato' in item_fullnames:
    possible_subtasks.append(('Merge', 'Lettuce-Tomato', 'Plate'))

  possible_delivery = []
  if 'Plate-ChoppedTomato' in item_fullnames:
    if chopped_lettuce:
      possible_subtasks.append(('Merge', 'Lettuce', 'Plate-Tomato'))
    possible_subtasks.append(('Deliver', 'Plate-Tomato'))
    possible_delivery.append(('Deliver', 'Plate-Tomato'))

  if 'ChoppedLettuce-Plate' in item_fullnames:
    if chopped_tomato:
      possible_subtasks.append(('Merge', 'Tomato', 'Lettuce-Plate'))
    possible_subtasks.append(('Deliver', 'Lettuce-Plate'))
    possible_delivery.append(('Deliver', 'Lettuce-Plate'))

  if 'ChoppedLettuce-Plate-ChoppedTomato' in item_fullnames:
    possible_subtasks.append(('Deliver', 'Lettuce-Plate-Tomato'))
    possible_delivery.append(('Deliver', 'Lettuce-Plate-Tomato'))

  # retain only possible subtasks in all_subtasks
  # todo_subtasks = {}
  # for subtask in all_subtasks:
  #     if subtask.name == 'Merge':
  #         tup1 = (subtask.name, subtask.args[0], subtask.args[1])
  #         tup2 = (subtask.name, subtask.args[1], subtask.args[0])
  #         if tup1 in possible_subtasks:
  #             todo_subtasks[tup1] = subtask
  #         elif tup2 in possible_subtasks:
  #             todo_subtasks[tup2] = subtask
  #     else:
  #         tup1 = (subtask.name, subtask.args[0])
  #         if tup1 in possible_subtasks:
  #             todo_subtasks[tup1] = subtask
  todo_subtasks = []
  for subtask_obj in all_subtasks:
    subtask_name = subtask_obj.name
    subtask_args = subtask_obj.args
    if subtask_name == 'Merge':
      tup1 = (subtask_name, subtask_args[0], subtask_args[1])
      tup2 = (subtask_name, subtask_args[1], subtask_args[0])
      if tup1 in possible_subtasks:
        todo_subtasks.append(tup1)
      elif tup2 in possible_subtasks:
        todo_subtasks.append(tup2)
    else:
      tup1 = (subtask_name, subtask_args[0])
      if tup1 in possible_subtasks:
        todo_subtasks.append(tup1)

  for subt in todo_subtasks:
    if subt in [('Merge', 'Lettuce', 'Plate-Tomato'),
                ('Merge', 'Lettuce-Tomato', 'Plate'),
                ('Merge', 'Lettuce-Tomato', 'Plate')]:
      mtp = ('Merge', 'Tomato', 'Plate')
      mlp = ('Merge', 'Lettuce', 'Plate')
      mtl = ('Merge', 'Tomato', 'Lettuce')
      if mtp in todo_subtasks:
        todo_subtasks.remove(mtp)
      if mlp in todo_subtasks:
        todo_subtasks.remove(mlp)
      if mtl in todo_subtasks:
        todo_subtasks.remove(mtl)

  if len(todo_subtasks) == 0:
    if len(possible_delivery) > 0:
      todo_subtasks = possible_delivery
    else:
      return list_agent_subtasks

  def is_reachable_item(world, agent, list_hold_agents, list_hold_gs):
    # check if reachable to the ingredient
    # (either agent is holding the ingredient or ingredient is on the reachable counter)
    reachable_to_item = False
    for hold_agent in list_hold_agents:
      if agent.name == hold_agent.name:
        reachable_to_item = True
        break

    if not reachable_to_item:
      if len(list_hold_gs) > 0 and reachable(world, agent, list_hold_gs):
        reachable_to_item = True

    return reachable_to_item

  # check if subtasks can be done alone
  reachabilities = {}
  for key in todo_subtasks:
    subtask_name = key[0]
    if subtask_name == 'Chop':
      item_name = key[1]
      agent_reachability = []
      for agent in obs.sim_agents:
        reachable_to_ingredient = is_reachable_item(
            obs.world, agent, item_2_hold_agents[item_name],
            item_2_hold_gs[item_name])
        reachable_to_cutboard = reachable(obs.world, agent, list_cutboards)
        agent_reachability.append(
            (reachable_to_ingredient, reachable_to_cutboard))
      reachabilities[key] = agent_reachability

    elif subtask_name == 'Merge':
      item_name1 = key[1]
      item_name2 = key[2]
      agent_reachability = []
      for agent in obs.sim_agents:
        reachable_to_item1 = is_reachable_item(obs.world, agent,
                                               item_2_hold_agents[item_name1],
                                               item_2_hold_gs[item_name1])
        reachable_to_item2 = is_reachable_item(obs.world, agent,
                                               item_2_hold_agents[item_name2],
                                               item_2_hold_gs[item_name2])
        agent_reachability.append((reachable_to_item1, reachable_to_item2))
      reachabilities[key] = agent_reachability

    elif subtask_name == "Deliver":
      item_name = key[1]
      agent_reachability = []
      for agent in obs.sim_agents:
        reachable_to_item = is_reachable_item(obs.world, agent,
                                              item_2_hold_agents[item_name],
                                              item_2_hold_gs[item_name])
        reachable_to_delivery = reachable(obs.world, agent, list_delivery)
        agent_reachability.append((reachable_to_item, reachable_to_delivery))
      reachabilities[key] = agent_reachability

  # check if current subtasks are valid. if valid, keep them
  workforce_shortage = {}
  random.shuffle(todo_subtasks)  # add randomness
  subtask_0 = todo_subtasks[0]
  for i_a, subtask in enumerate(list_agent_subtasks):
    # if invalid subtask, remove it
    if subtask is not None and subtask not in todo_subtasks:
      list_agent_subtasks[i_a] = None

    if list_agent_subtasks[i_a] is not None:
      # check if the agent can do the subtask alone
      if all(reachabilities[subtask][i_a]):
        # remove it from todo_subtasks as it is already assigned
        # --> this will automatically set the subtask of the next agents who are holding this to None.
        todo_subtasks.remove(subtask)
        # if there are the previous agents who are assigned with this subtask but who can't do this alone, remove their subtasks as well.
        if subtask in workforce_shortage:
          for i_b in workforce_shortage[subtask]:
            list_agent_subtasks[i_b] = None
          del workforce_shortage[subtask]
      else:
        workforce_shortage[subtask] = workforce_shortage.get(subtask,
                                                             []) + [i_a]

  # no subtasks left
  if len(todo_subtasks) == 0:
    for i_a, subtask in enumerate(list_agent_subtasks):
      if subtask is None:
        list_agent_subtasks[i_a] = subtask_0

  # if all agents are assigned with subtasks, return
  if all([subtask is not None for subtask in list_agent_subtasks]):
    return list_agent_subtasks

  # assign the subtask currently in workforce shortage to other agents who don't have any subtask
  n_subtasks_shortage = len(workforce_shortage.keys())
  if n_subtasks_shortage > 0:
    # for now, just assign one to all
    assigned_subtask = random.choice(list(workforce_shortage.keys()))
    for i_a, subtask in enumerate(list_agent_subtasks):
      if subtask is None:
        list_agent_subtasks[i_a] = assigned_subtask

    return list_agent_subtasks

  # assign sole subtasks first
  subtask_0 = todo_subtasks[0]
  for i_a, subtask in enumerate(list_agent_subtasks):
    if subtask is not None:
      continue

    # find a subtask that can be done alone
    for subtask_assign in todo_subtasks:
      if all(reachabilities[subtask_assign][i_a]):
        list_agent_subtasks[i_a] = subtask_assign
        todo_subtasks.remove(subtask_assign)
        break

  # no subtasks left
  if len(todo_subtasks) == 0:
    for i_a, subtask in enumerate(list_agent_subtasks):
      if subtask is None:
        list_agent_subtasks[i_a] = subtask_0

  # if all agents are assigned with subtasks, return
  if all([subtask is not None for subtask in list_agent_subtasks]):
    return list_agent_subtasks

  # if no subtask can be done alone, assign the first subtask in the list
  subtask_0 = todo_subtasks[0]
  for i_a, subtask in enumerate(list_agent_subtasks):
    if subtask is None:
      list_agent_subtasks[i_a] = subtask_0

  return list_agent_subtasks


def select_action_given_subtask(i_a, obs, subtask):
  dict_info = get_world_info(obs)
  item_fullnames = dict_info["item_fullnames"]
  item_2_hold_agents = dict_info["item_2_hold_agents"]
  item_2_hold_gs = dict_info["item_2_hold_gs"]
  list_cutboards = dict_info["list_cutboards"]
  list_delivery = dict_info["list_delivery"]
  list_emptycounters = dict_info["list_emptycounters"]

  if len(item_fullnames) == 0:
    possible_actions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
    return random.choice(possible_actions)

  agent = obs.sim_agents[i_a]
  others_pos = []
  for i_o, oth in enumerate(obs.sim_agents):
    if i_a != i_o:
      others_pos.append(oth.location)

  ME = 'me'
  OTHER = 'other'

  def get_item_loc(item):
    loc = None
    for agt in item_2_hold_agents[item]:
      if agent.name == agt.name:
        loc = ME
      else:
        loc = OTHER
      break

    if loc is None:
      for obj in item_2_hold_gs[item]:
        return obj

    return loc

  def get_opt_acts(agent_loc, obj_loc, obj_nextto_loc):
    x_a, y_a = agent_loc
    x_no, y_no = obj_nextto_loc

    if x_a == x_no and y_a == y_no:
      x_o, y_o = obj_loc
      opt_acts = [(x_o - x_no, y_o - y_no)]
    else:
      d_x = x_no - x_a
      d_y = y_no - y_a
      # move to loc
      opt_acts = []
      if d_x < 0:
        opt_acts.append((-1, 0))
      elif d_x > 0:
        opt_acts.append((1, 0))
      if d_y < 0:
        opt_acts.append((0, -1))
      elif d_y > 0:
        opt_acts.append((0, 1))

    final_acts = []
    for act in opt_acts:
      if (x_a + act[0], y_a + act[1]) not in others_pos:
        final_acts.append(act)

    if len(final_acts) == 0:
      final_acts = None

    return final_acts

  opt_acts = None
  if subtask[0] == 'Chop':
    # check if reachable to item
    item1 = subtask[1]

    # find item location
    loc = get_item_loc(item1)

    # holding wrong item
    if agent.holding is not None and agent.holding.name != item1:
      path, targeted_objs = find_path(obs.world, agent, list_emptycounters)
      opt_acts = get_opt_acts(agent.location, targeted_objs[0].location,
                              path[-1])
    elif loc is None or loc == OTHER:
      # do nothing (random action)
      pass
    elif loc == ME:
      # go to cutboard
      path, targeted_objs = find_path(obs.world, agent, list_cutboards)
      if len(path) > 0:
        opt_acts = get_opt_acts(agent.location, targeted_objs[0].location,
                                path[-1])
      else:
        # TODO: find both reachable counters
        pass
    else:
      path, targeted_objs = find_path(obs.world, agent, [loc])
      if len(path) > 0:
        opt_acts = get_opt_acts(agent.location, targeted_objs[0].location,
                                path[-1])
      else:
        # wait
        pass

  elif subtask[0] == 'Merge':
    item1 = subtask[1]
    item2 = subtask[2]
    loc_1 = get_item_loc(item1)
    loc_2 = get_item_loc(item2)

    if agent.holding is not None and agent.holding.name not in [item1, item2]:
      path, targeted_objs = find_path(obs.world, agent, list_emptycounters)
      opt_acts = get_opt_acts(agent.location, targeted_objs[0].location,
                              path[-1])
    elif loc_1 is not None and loc_1 == ME:
      # go to item2
      if isinstance(loc_2, GridSquare):
        path, targeted_objs = find_path(obs.world, agent, [loc_2])
        if len(path) > 0:
          opt_acts = get_opt_acts(agent.location, targeted_objs[0].location,
                                  path[-1])
      elif loc_2 == OTHER:
        path, targeted_objs = find_path(obs.world, agent, list_emptycounters)
        opt_acts = get_opt_acts(agent.location, targeted_objs[0].location,
                                path[-1])
    elif loc_2 is not None and loc_2 == ME:
      # go to item1
      if isinstance(loc_1, GridSquare):
        path, targeted_objs = find_path(obs.world, agent, [loc_1])
        if len(path) > 0:
          opt_acts = get_opt_acts(agent.location, targeted_objs[0].location,
                                  path[-1])
      elif loc_1 == OTHER:
        path, targeted_objs = find_path(obs.world, agent, list_emptycounters)
        opt_acts = get_opt_acts(agent.location, targeted_objs[0].location,
                                path[-1])
    else:
      list_obj = []
      if isinstance(loc_1, GridSquare):
        list_obj.append(loc_1)
      if isinstance(loc_2, GridSquare):
        list_obj.append(loc_2)

      if len(list_obj) == 2:
        path, targeted_objs = find_path(obs.world, agent, list_obj)
        if len(path) > 0:
          opt_acts = get_opt_acts(agent.location, targeted_objs[0].location,
                                  path[-1])
  elif subtask[0] == 'Deliver':
    item1 = subtask[1]

    # find item location
    loc = get_item_loc(item1)

    # holding wrong item
    if agent.holding is not None and agent.holding.name != item1:
      path, targeted_objs = find_path(obs.world, agent, list_emptycounters)
      opt_acts = get_opt_acts(agent.location, targeted_objs[0].location,
                              path[-1])
    elif loc is None or loc == OTHER:
      # do nothing (random action)
      pass
    elif loc == ME:
      # go to deliver
      path, targeted_objs = find_path(obs.world, agent, list_delivery)
      if len(path) > 0:
        opt_acts = get_opt_acts(agent.location, targeted_objs[0].location,
                                path[-1])
      else:
        # TODO: find both reachable counters
        pass
    else:
      path, targeted_objs = find_path(obs.world, agent, [loc])
      if len(path) > 0:
        opt_acts = get_opt_acts(agent.location, targeted_objs[0].location,
                                path[-1])
      else:
        # wait
        pass

  eps = 0.1
  if opt_acts is None or random.random() < eps:
    possible_actions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
    return random.choice(possible_actions)

  return random.choice(opt_acts)


def subtask_2_index(subtask):

  if subtask == ('Chop', 'Tomato'):
    return 0

  if subtask == ('Chop', 'Lettuce'):
    return 1

  if subtask == ('Merge', 'Tomato', 'Plate'):
    return 2

  if subtask == ('Merge', 'Lettuce', 'Plate'):
    return 3

  if subtask == ('Merge', 'Tomato', 'Lettuce'):
    return 4

  if subtask in [('Merge', 'Lettuce-Tomato', 'Plate'),
                 ('Merge', 'Lettuce', 'Plate-Tomato'),
                 ('Merge', 'Tomato', 'Lettuce-Plate')]:
    return 5

  if subtask in [('Deliver', 'Plate-Tomato'), ('Deliver', 'Lettuce-Plate'),
                 ('Deliver', 'Lettuce-Plate-Tomato')]:
    return 6

  raise ValueError('Invalid subtask')


def generate_data(level, n_traj):

  expert_trajs = defaultdict(list)
  env = Overcooked(level)

  env.reset()
  n_agents = env.num_agents

  list_total_reward = []
  for _ in range(n_traj):
    obs, infos = env.reset()
    episode_reward = {aname: 0 for aname in env.agents}

    list_agent_subtasks = [None for _ in range(n_agents)]
    total_reward = 0
    samples = []
    while True:
      actions = {}
      latents = {}
      raw_env_obs = infos[env.agents[0]]["env_new"]
      list_agent_subtasks = assign_subtasks(raw_env_obs, list_agent_subtasks)

      for aname in env.agents:
        subtask = list_agent_subtasks[env.agent_name_mapping[aname]]
        latents[aname] = subtask_2_index(subtask)
        action = select_action_given_subtask(env.agent_name_mapping[aname],
                                             raw_env_obs, subtask)
        action = env.action_mapping[action]
        actions[aname] = action

      next_obs, rewards, dones, truncs, infos = env.step(actions)
      samples.append((obs, actions, next_obs, latents, rewards, dones))

      for aname in env.agents:
        episode_reward[aname] += rewards[aname]
        total_reward += rewards[aname]

      obs = next_obs

      if all(truncs.values()) or all(dones.values()):
        break

    list_total_reward.append(total_reward)
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
  return expert_trajs


if __name__ == "__main__":
  cur_dir = os.path.dirname(__file__)

  level = 'open-divider_salad'
  level = 'open-divider_tl'

  n_subtasks = 7

  n_traj = 50
  n_sal = 50
  n_tl = n_traj - n_sal

  trajs1 = generate_data('open-divider_salad', n_sal)
  trajs2 = generate_data('open-divider_tl', n_tl)

  trajs_merged = {}
  for key in trajs1.keys():
    trajs_merged[key] = trajs1.get(key, []) + trajs2.get(key, [])

  save_dir = cur_dir
  if save_dir is not None:
    file_path = os.path.join(save_dir,
                             f"overcooked-{n_sal}-{n_tl}_{n_traj}.pkl")
    with open(file_path, 'wb') as f:
      pickle.dump(trajs_merged, f)

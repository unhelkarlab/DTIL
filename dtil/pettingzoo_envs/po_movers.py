import functools

import gymnasium
from gymnasium.spaces import Discrete
from gymnasium.utils import seeding
import numpy as np

from pettingzoo import ParallelEnv
from aic_domain.box_push_v2.mdp import MDP_Movers_Task
from aic_domain.box_push_v2.maps import MAP_MOVERS
from aic_core.utils.mdp_utils import StateSpace
from aic_domain.box_push.define import BoxState


class PO_Movers(ParallelEnv):
  metadata = {"render_modes": ["human"], "name": "ma_movers"}

  def __init__(self, render_mode=None):
    """
        The init method takes in environment arguments and should define 
        the following attributes:
        - possible_agents
        - render_mode
    """

    game_map = MAP_MOVERS
    self.max_step = 150

    self.mmdp = MDP_Movers_Task(**game_map)
    self.possible_agents = [idx for idx in range(self.mmdp.num_action_factors)]
    self._init_obs_spaces()

    init_bstate = [0] * len(self.mmdp.boxes)
    a1_init = game_map["a1_init"]
    a2_init = game_map["a2_init"]

    self.init_state = self.mmdp.conv_sim_states_to_mdp_sidx(
        [init_bstate, a1_init, a2_init])

    for agent in self.possible_agents:
      self.action_spaces[agent] = Discrete(self.mmdp.list_num_actions[agent])
      self.observation_spaces[agent] = Discrete(self.num_obs)

    # optional: a mapping between agent name and ID
    self.agent_name_mapping = dict(
        zip(self.possible_agents, list(range(len(self.possible_agents)))))
    self.render_mode = render_mode

  def _init_obs_spaces(self):
    self.agent_position_space = self.mmdp.pos1_space

    list_rel_pos = ["None"]
    for rel_pos_x in [-1, 0, 1]:
      for rel_pos_y in [-1, 0, 1]:
        list_rel_pos.append((rel_pos_x, rel_pos_y))

    self.teammate_position_space = StateSpace(list_rel_pos)

    self.dict_factored_obs_space = {
        0: self.agent_position_space,
        1: self.teammate_position_space
    }

    list_box_positions = ["unknown", "original", "with_us", "goal"]
    for idx in range(len(self.mmdp.boxes)):
      self.dict_factored_obs_space[idx + 2] = StateSpace(list_box_positions)

    # give index to each observation
    self.num_obs_factors = len(self.dict_factored_obs_space)
    self.list_num_obs = []
    for idx in range(self.num_obs_factors):
      self.list_num_obs.append(self.dict_factored_obs_space.get(idx).num_states)

    self.num_obs = np.prod(self.list_num_obs)
    np_list_idx = np.arange(self.num_obs, dtype=np.int32)
    self.np_factered_idx_to_obs_idx = np_list_idx.reshape(self.list_num_obs)
    np_obs_idx_to_factored_idx = np.zeros((self.num_obs, self.num_obs_factors),
                                          dtype=np.int32)
    for obs, idx in np.ndenumerate(self.np_factered_idx_to_obs_idx):
      np_obs_idx_to_factored_idx[idx] = obs
    self.np_obs_idx_to_factored_idx = np_obs_idx_to_factored_idx

  def conv_state_to_obs(self, state_idx, joint_prev_action=None):
    len_s_space = len(self.mmdp.dict_factored_statespace)
    state_vec = self.mmdp.conv_idx_to_state(state_idx)

    pos1 = self.mmdp.pos1_space.idx_to_state[state_vec[0]]
    pos2 = self.mmdp.pos2_space.idx_to_state[state_vec[1]]

    box_states = []
    for idx in range(2, len_s_space):
      box_sidx = state_vec[idx]
      box_state = self.mmdp.dict_factored_statespace[idx].idx_to_state[box_sidx]
      box_states.append(box_state)

    def get_is_neighboring_and_rel_pos(pos_me, pos_target):
      x = pos_target[0] - pos_me[0]
      y = pos_target[1] - pos_me[1]

      rel_pos = (x, y)
      is_neighboring = (x in [-1, 0, 1] and y in [-1, 0, 1])

      return is_neighboring, rel_pos

    def get_teammate_idx(pos_me, pos_mate):
      is_neighboring, rel_pos = get_is_neighboring_and_rel_pos(pos_me, pos_mate)
      if is_neighboring:
        return self.teammate_position_space.state_to_idx[rel_pos]
      else:
        return self.teammate_position_space.state_to_idx["None"]

    def get_box_idxs(pos, box_states):
      is_goal_neigh, _ = get_is_neighboring_and_rel_pos(pos, self.mmdp.goals[0])

      list_box_idxs = []
      for idx in range(len(self.mmdp.boxes)):
        is_box_neigh, _ = get_is_neighboring_and_rel_pos(
            pos, self.mmdp.boxes[idx])

        bst = box_states[idx][0]

        box_aware = None
        if bst == BoxState.Original:
          if is_box_neigh:
            box_aware = "original"
          else:
            box_aware = "unknown"
        elif bst == BoxState.WithBoth:
          box_aware = "with_us"
        elif bst == BoxState.OnGoalLoc:
          if is_goal_neigh:
            box_aware = "goal"
          else:
            box_aware = "unknown"
        else:
          raise ValueError(f"Unknown box state: {bst}")

        list_box_idxs.append(
            self.dict_factored_obs_space[idx + 2].state_to_idx[box_aware])

      return tuple(list_box_idxs)

    def get_obs_idx(pos1, pos2, b_states):
      pos_idx = self.agent_position_space.state_to_idx[pos1]
      mate_idx = get_teammate_idx(pos1, pos2)
      box_idxs = get_box_idxs(pos1, box_states)

      return self.np_factered_idx_to_obs_idx[(pos_idx, mate_idx, *box_idxs)]

    dict_obs = {}
    dict_obs[self.possible_agents[0]] = get_obs_idx(pos1, pos2, box_states)
    dict_obs[self.possible_agents[1]] = get_obs_idx(pos2, pos1, box_states)

    if joint_prev_action is not None:
      dict_aux = {}
      if get_is_neighboring_and_rel_pos(pos1, pos2)[0]:
        aux1 = aux2 = [
            joint_prev_action[self.possible_agents[0]],
            joint_prev_action[self.possible_agents[1]]
        ]
      else:
        aux1 = [
            joint_prev_action[self.possible_agents[0]],
            self.mmdp.list_num_actions[self.possible_agents[1]]
        ]
        aux2 = [
            self.mmdp.list_num_actions[self.possible_agents[0]],
            joint_prev_action[self.possible_agents[1]]
        ]

      dict_aux[self.possible_agents[0]] = np.hstack(aux1)
      dict_aux[self.possible_agents[1]] = np.hstack(aux2)
    else:
      dict_aux = None

    return dict_obs, dict_aux

  @functools.lru_cache(maxsize=None)
  def observation_space(self, agent):
    # gymnasium spaces are defined and documented here:
    #    https://gymnasium.farama.org/api/spaces/
    return self.observation_spaces[agent]

  @functools.lru_cache(maxsize=None)
  def action_space(self, agent):
    return self.action_spaces[agent]

  def render(self):
    if self.render_mode is None:
      gymnasium.logger.warn(
          "You are calling render method without specifying any render mode.")
      return
    pass

  def close(self):
    pass

  def _seed(self, seed=None):
    self.np_random, seed = seeding.np_random(seed)

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
    self.cur_step = 0

    self.cur_state = self.init_state
    self.cur_auxiliary_inputs = {agent: None for agent in self.agents}

    observations, _ = self.conv_state_to_obs(self.cur_state)
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

    list_actions = [actions[agent] for agent in self.agents]
    aidx = self.mmdp.conv_action_to_idx(tuple(list_actions))

    infos = {agent: {} for agent in self.agents}

    if aidx not in self.mmdp.legal_actions(self.cur_state):
      obs, aux = self.conv_state_to_obs(self.cur_state, actions)
      self.cur_auxiliary_inputs = aux
      rewards = {agent: -10000 for agent in self.agents}
      dones = {agent: False for agent in self.agents}
      truncs = {agent: trunc for agent in self.agents}
      return obs, rewards, dones, truncs, infos

    self.cur_state = self.mmdp.transition(self.cur_state, aidx)
    obs, aux = self.conv_state_to_obs(self.cur_state, actions)
    self.cur_auxiliary_inputs = aux

    if self.mmdp.is_terminal(self.cur_state):
      dones = {agent: True for agent in self.agents}
      truncs = {agent: False for agent in self.agents}
    else:
      dones = {agent: False for agent in self.agents}
      truncs = {agent: trunc for agent in self.agents}

    rewards = {agent: -1 for agent in self.agents}

    return obs, rewards, dones, truncs, infos

  def get_auxiliary_obs(self, agent):
    return self.cur_auxiliary_inputs[agent]

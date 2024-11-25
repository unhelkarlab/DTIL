import functools

import gymnasium
from gymnasium.spaces import Discrete
from gymnasium.utils import seeding
import numpy as np

from pettingzoo import ParallelEnv
from aic_domain.rescue.mdp import MDP_Rescue_Task
from aic_domain.rescue.maps import MAP_RESCUE
from aic_domain.rescue import Location, E_Type, is_work_done
from aic_core.utils.mdp_utils import StateSpace


class Rescue(ParallelEnv):
  metadata = {"render_modes": ["human"], "name": "ma_rescue"}

  def __init__(self, render_mode=None):
    """
        The init method takes in environment arguments and should define 
        the following attributes:
        - possible_agents
        - render_mode
    """

    game_map = MAP_RESCUE
    self.max_step = 30

    self.mmdp = MDP_Rescue_Task(**game_map)
    self.possible_agents = [idx for idx in range(self.mmdp.num_action_factors)]
    self._init_obs_spaces()

    init_wstate = [1] * len(self.mmdp.work_locations)
    a1_init = game_map["a1_init"]
    a2_init = game_map["a2_init"]

    self.init_state = self.mmdp.conv_sim_states_to_mdp_sidx(
        (init_wstate, a1_init, a2_init))

    for agent in self.possible_agents:
      self.action_spaces[agent] = Discrete(self.mmdp.list_num_actions[agent])
      self.observation_spaces[agent] = Discrete(self.num_obs)

    # optional: a mapping between agent name and ID
    self.agent_name_mapping = dict(
        zip(self.possible_agents, list(range(len(self.possible_agents)))))
    self.render_mode = render_mode

  def _init_obs_spaces(self):
    self.agent_position_space = self.mmdp.pos1_space

    self.reachable_landmarks = []
    for key, val in self.mmdp.connections.items():
      if len(val) != 0:
        self.reachable_landmarks.append(Location(E_Type.Place, key))

    list_teammate_pos_space = ["unknown", "with_me"] + self.reachable_landmarks
    print(list_teammate_pos_space)

    self.teammate_position_space = StateSpace(list_teammate_pos_space)
    self.work_states_space = self.mmdp.work_states_space

    self.dict_factored_obs_space = {
        0: self.agent_position_space,
        1: self.teammate_position_space,
        2: self.work_states_space
    }

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
    state_vec = self.mmdp.conv_idx_to_state(state_idx)

    pos1 = self.mmdp.pos1_space.idx_to_state[state_vec[0]]
    pos2 = self.mmdp.pos2_space.idx_to_state[state_vec[1]]
    work_states = self.mmdp.work_states_space.idx_to_state[state_vec[2]]

    def get_mate_pos(pos_me, pos_mate):
      # together
      if pos_me[0] == pos_mate[0] and pos_me[1] == pos_mate[1]:
        return "with_me"
      elif pos_mate in self.reachable_landmarks:
        return pos_mate
      else:
        return "unknown"

    def get_obs_idx(pos_me, pos_mate, w_states):
      pos_idx = self.agent_position_space.state_to_idx[pos_me]
      mate_idx = self.teammate_position_space.state_to_idx[get_mate_pos(
          pos_me, pos_mate)]
      w_states_idx = self.work_states_space.state_to_idx[w_states]

      return self.np_factered_idx_to_obs_idx[(pos_idx, mate_idx, w_states_idx)]

    dict_obs = {}
    dict_obs[self.possible_agents[0]] = get_obs_idx(pos1, pos2, work_states)
    dict_obs[self.possible_agents[1]] = get_obs_idx(pos2, pos1, work_states)

    if joint_prev_action is not None:
      dict_aux = {}
      if get_mate_pos(pos1, pos2) != "unknown":
        aux1 = [
            joint_prev_action[self.possible_agents[0]],
            joint_prev_action[self.possible_agents[1]]
        ]
      else:
        aux1 = [
            joint_prev_action[self.possible_agents[0]],
            self.mmdp.list_num_actions[self.possible_agents[1]]
        ]

      if get_mate_pos(pos2, pos1) != "unknown":
        aux2 = [
            joint_prev_action[self.possible_agents[0]],
            joint_prev_action[self.possible_agents[1]]
        ]
      else:
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
    self.score = 0

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

    # reward
    work_states, _, _ = self.mmdp.conv_mdp_sidx_to_sim_states(self.cur_state)
    rescued_place = []
    for idx in range(len(work_states)):
      if is_work_done(idx, work_states, self.mmdp.work_info[idx].coupled_works):
        place_id = self.mmdp.work_info[idx].rescue_place
        if place_id not in rescued_place:
          rescued_place.append(place_id)

    score = 0
    for place_id in rescued_place:
      score += self.mmdp.places[place_id].helps

    score_increment = score - self.score
    self.score = score

    rewards = {agent: score_increment for agent in self.agents}

    return obs, rewards, dones, truncs, infos

  def get_auxiliary_obs(self, agent):
    return self.cur_auxiliary_inputs[agent]

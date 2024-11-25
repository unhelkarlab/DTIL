import os
import pickle
import functools
from collections import defaultdict
import gymnasium
from gymnasium.spaces import Discrete, Box
from gymnasium.utils import seeding
import numpy as np
from pettingzoo import ParallelEnv

from aic_domain.rescue.agent import AIAgent_Rescue_PartialObs
from aic_domain.rescue.maps import MAP_RESCUE
from aic_domain.rescue.policy import Policy_Rescue
from aic_domain.rescue.mdp import MDP_Rescue_Agent, MDP_Rescue_Task
from aic_domain.rescue import (Location, E_Type, get_score, AGENT_ACTIONSPACE)


class PO_Flood_V2(ParallelEnv):
  metadata = {"render_modes": ["human"], "name": "PO_Flood-v2"}
  SAME_LOC = "SAME_LOCATION"

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
    self._init_obs_act_spaces()

    init_states = ([1] * len(game_map["work_locations"]), game_map["a1_init"],
                   game_map["a2_init"])
    self.init_state = self.mmdp.conv_sim_states_to_mdp_sidx(init_states)

    # optional: a mapping between agent name and ID
    self.agent_name_mapping = dict(
        zip(self.possible_agents, list(range(len(self.possible_agents)))))
    self.render_mode = render_mode

  def _init_obs_act_spaces(self):
    '''
    n_lm: num accessible landmarks = 6
    my_loc, my_a, fr_o, fr_loc,    fr_a, w1, w2, w3, w4
    32      6     1     n_lm + 1   6     1   1   1   1
    '''
    mdp = self.mmdp

    self.possible_locations = [self.SAME_LOC]
    for place_id in mdp.connections:
      if len(mdp.connections[place_id]) > 0:
        self.possible_locations.append(Location(E_Type.Place, place_id))

    n_fr_locations = len(self.possible_locations)
    self.dict_location_to_idx = {
        self.possible_locations[idx]: idx
        for idx in range(n_fr_locations)
    }

    # NOTE: num_locations are identical for two agents
    num_loc = mdp.pos1_space.num_states  # = 32

    num_obs_bits = (num_loc + sum(mdp.list_num_actions) + 1 + n_fr_locations +
                    len(mdp.work_locations))
    self.observation_spaces = {
        aname: Box(low=np.zeros(num_obs_bits), high=np.ones(num_obs_bits))
        for aname in self.possible_agents
    }
    self.action_spaces = {
        self.possible_agents[idx]: Discrete(mdp.list_num_actions[idx])
        for idx in range(len(self.possible_agents))
    }

  def conv_state_to_obs(self, state_idx, joint_prev_action=None):
    '''
    my_loc, my_a, fr_o, fr_loc,    fr_a, w1, w2, w3, w4
    32      6     1     n_lm + 1   6     1   1   1   1
    '''

    def one_hot(n, dim):
      np_oh = np.zeros(dim)
      np_oh[n] = 1
      return np_oh

    mdp = self.mmdp
    state_vec = mdp.conv_idx_to_state(state_idx)
    w_state = mdp.work_states_space.idx_to_state[state_vec[2]]

    def get_obs(aidx):
      fidx = 1 - aidx

      my_locidx = state_vec[aidx]
      my_loc = mdp.dict_factored_statespace[aidx].idx_to_state[my_locidx]
      fr_loc = mdp.dict_factored_statespace[fidx].idx_to_state[state_vec[fidx]]

      fr_o = 0  # assume unobserved at first
      # unless observed, fr_locidx and fr_a are set to 0 (like a dummy value)
      fr_locidx = 0
      fr_a = 0
      if fr_loc in self.dict_location_to_idx:
        fr_o = 1
        fr_locidx = self.dict_location_to_idx[fr_loc]
      elif my_loc == fr_loc:
        fr_o = 1
        fr_locidx = self.dict_location_to_idx[self.SAME_LOC]

      if fr_o == 1:
        fr_locidx = one_hot(fr_locidx, len(self.possible_locations))
        fr_a = one_hot(joint_prev_action[self.possible_agents[fidx]],
                       self.action_spaces[fidx].n)
      else:
        fr_locidx = np.zeros(len(self.possible_locations))
        fr_a = np.zeros(self.action_spaces[fidx].n)

      my_locidx = one_hot(my_locidx, mdp.pos1_space.num_states)
      my_a = one_hot(joint_prev_action[self.possible_agents[aidx]],
                     self.action_spaces[aidx].n)

      fr_o = np.array([fr_o])
      wstate = np.array(w_state)

      list_obs_factors = [my_locidx, my_a, fr_o, fr_locidx, fr_a, wstate]
      return np.hstack(list_obs_factors)

    a1name = self.possible_agents[0]
    a2name = self.possible_agents[1]
    dict_obs = {}
    dict_obs[a1name] = get_obs(0)
    dict_obs[a2name] = get_obs(1)

    return dict_obs

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

    actions = {aname: 4 for aname in self.agents}  # 4 implies "stay"
    observations = self.conv_state_to_obs(self.cur_state, actions)
    infos = {agent: {} for agent in self.agents}

    return observations, infos

  def _get_score(self, state_idx):
    mdp = self.mmdp
    work_states, _, _ = mdp.conv_mdp_sidx_to_sim_states(state_idx)

    return get_score(work_states, mdp.work_info, mdp.places)

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

    prev_score = self._get_score(self.cur_state)

    self.cur_state = self.mmdp.transition(self.cur_state, aidx)
    obs = self.conv_state_to_obs(self.cur_state, actions)

    cur_score = self._get_score(self.cur_state)

    if self.mmdp.is_terminal(self.cur_state):
      dones = {agent: True for agent in self.agents}
      truncs = {agent: False for agent in self.agents}
    else:
      dones = {agent: False for agent in self.agents}
      truncs = {agent: trunc for agent in self.agents}

    rewards = {agent: cur_score - prev_score for agent in self.agents}

    return obs, rewards, dones, truncs, infos


class PO_Flood_AIAgent(AIAgent_Rescue_PartialObs):

  def __init__(self, init_tup_states, agent_idx, policy_model,
               possible_locations) -> None:
    super().__init__(init_tup_states, agent_idx, policy_model, True)
    self.possible_locations = possible_locations
    self.PREV_LATENT = None
    self.PREV_AUX = None

  def choose_action(self, obs, prev_latent, sample=False, **kwargs):
    if prev_latent == self.PREV_LATENT:
      self.init_latent(obs)
    else:
      self.update_mental_state(None, None, obs)

    latent = self.get_current_latent()
    action = self.get_action(obs)

    latent_idx = self.conv_latent_to_idx(latent)
    action_idx = self.conv_action_to_idx((action, ))[0]
    return latent_idx, action_idx

  # replace virtual functions
  def observed_states(self, obs):
    # TODO: don't directly use numbers for indexing
    my_loc = np.where(obs[0:32] == 1)[0][0]
    my_a = np.where(obs[32:38] == 1)[0][0]
    fr_o = bool(obs[38])
    wstate = obs[52:56]

    mdp = self.agent_model.get_reference_mdp()
    asm_work_state = tuple(wstate)

    possible_asm_loc = [
        Location(E_Type.Route, 1, 0),
        Location(E_Type.Route, 1, 1)
    ]

    def get_agent_locs(aidx):
      asm_my_pos = mdp.dict_factored_statespace[aidx].idx_to_state[my_loc]
      if fr_o:
        fr_loc = np.where(obs[39:46] == 1)[0][0]
        if self.possible_locations[fr_loc] == PO_Flood_V2.SAME_LOC:  # same
          asm_fr_pos = asm_my_pos
        else:
          asm_fr_pos = self.possible_locations[fr_loc]
      else:
        for asm_loc in possible_asm_loc:
          if asm_my_pos != asm_loc:
            asm_fr_pos = asm_loc
            break
      return asm_my_pos, asm_fr_pos

    if self.agent_idx == 0:
      asm_a1_pos, asm_a2_pos = get_agent_locs(self.agent_idx)
    else:
      asm_a2_pos, asm_a1_pos = get_agent_locs(self.agent_idx)

    return asm_work_state, asm_a1_pos, asm_a2_pos

  def observed_actions(self, tup_actions, obs) -> tuple:
    my_a = np.where(obs[32:38] == 1)[0][0]
    fr_o = bool(obs[38])

    observed_actions = [None, None]
    my_idx, fr_idx = (0, 1) if self.agent_idx == 0 else (1, 0)

    observed_actions[my_idx] = AGENT_ACTIONSPACE.idx_to_action[my_a]
    if fr_o:
      fr_a = np.where(obs[46:52] == 1)[0][0]
      observed_actions[fr_idx] = AGENT_ACTIONSPACE.idx_to_action[fr_a]

    return tuple(observed_actions)


def generate_data(save_dir, env_name, n_traj, render):
  expert_trajs = defaultdict(list)
  if env_name == "PO_Flood-v2":
    env = PO_Flood_V2(render_mode="human")
  else:
    raise NotImplementedError()

  TEMPERATURE = 0.3
  GAME_MAP = MAP_RESCUE
  MDP_TASK = MDP_Rescue_Task(**GAME_MAP)
  MDP_AGENT = MDP_Rescue_Agent(**GAME_MAP)
  POLICY_1 = Policy_Rescue(MDP_TASK, MDP_AGENT, TEMPERATURE, 0)
  POLICY_2 = Policy_Rescue(MDP_TASK, MDP_AGENT, TEMPERATURE, 1)

  init_states = ([1] * len(GAME_MAP["work_locations"]), GAME_MAP["a1_init"],
                 GAME_MAP["a2_init"])

  agents = {
      env.possible_agents[0]:
      PO_Flood_AIAgent(init_states, 0, POLICY_1, env.possible_locations),
      env.possible_agents[1]:
      PO_Flood_AIAgent(init_states, 1, POLICY_2, env.possible_locations)
  }

  list_total_reward = []
  for _ in range(n_traj):
    obs, infos = env.reset()
    episode_reward = {aname: 0 for aname in env.agents}

    for aname in agents:
      agents[aname].init_latent(obs[aname])

    total_reward = 0
    samples = []
    while True:
      actions = {}
      latents = {}
      for aname in env.agents:
        latent = agents[aname].get_current_latent()
        action = agents[aname].get_action(obs[aname])

        latents[aname] = agents[aname].conv_latent_to_idx(latent)
        # TODO: conv_action_to_idx need refactoring.
        #       make it accept single action (non-tuple) input as well
        actions[aname] = agents[aname].conv_action_to_idx((action, ))[0]

      next_obs, rewards, dones, truncs, infos = env.step(actions)

      samples.append((obs, actions, next_obs, latents, rewards, dones))

      for aname in env.agents:
        if not (all(truncs.values()) or all(dones.values())):
          agents[aname].update_mental_state(obs[aname], None, next_obs[aname])
        episode_reward[aname] += rewards[aname]
        total_reward += rewards[aname]

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

  traj = generate_data(cur_dir, "PO_Flood-v2", 100, False)

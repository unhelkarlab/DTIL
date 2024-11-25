import os
import pickle
import functools
from collections import defaultdict
from aic_core.models.policy import CachedPolicyInterface
import gymnasium
from gymnasium.spaces import Discrete, Box
from gymnasium.utils import seeding
import numpy as np

from pettingzoo import ParallelEnv
from aic_domain.box_push_v2.mdp import (MDP_Movers_Agent, MDP_Movers_Task)
from aic_domain.box_push_v2.maps import MAP_MOVERS
from aic_core.utils.mdp_utils import StateSpace
from aic_domain.box_push.define import BoxState
from aic_domain.box_push_v2.agent import BoxPushAIAgent_PO_Team
from aic_domain.box_push_v2.policy import Policy_Movers
from aic_domain.agent import AIAgent_PartialObs
from aic_domain.box_push_v2 import (conv_box_idx_2_state, conv_box_state_2_idx,
                                    BoxState, AGENT_ACTIONSPACE)


class PO_Movers_V2(ParallelEnv):
  metadata = {"render_modes": ["human"], "name": "PO_Movers-v2"}

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
    self._init_obs_act_spaces()

    init_bstate = [0] * len(self.mmdp.boxes)
    a1_init = game_map["a1_init"]
    a2_init = game_map["a2_init"]

    self.init_state = self.mmdp.conv_sim_states_to_mdp_sidx(
        [init_bstate, a1_init, a2_init])

    # optional: a mapping between agent name and ID
    self.agent_name_mapping = dict(
        zip(self.possible_agents, list(range(len(self.possible_agents)))))
    self.render_mode = render_mode

  def _init_obs_act_spaces(self):
    '''
    fr_o: whether friend is observed
    box state: None, BoxState.Original, BoxState.WithBoth, BoxState.OnGoalLoc
    fr_x: [-1, 0, 1] maps to [0, 1, 2] bits
    my_x, my_y, my_a, fr_o, fr_x, fr_y, fr_a, box1, box2, box3
    7,    7     6     1     3     3     6     4     4     4
    '''
    self.possible_box_state = [
        None, BoxState.Original, BoxState.WithBoth, BoxState.OnGoalLoc
    ]
    self.dict_boxstate_to_idx = {
        self.possible_box_state[idx]: idx
        for idx in range(len(self.possible_box_state))
    }
    mdp = self.mmdp
    num_state_bits = (mdp.x_grid + mdp.y_grid + sum(mdp.list_num_actions) + 1 +
                      3 * 2 + len(mdp.boxes) * 4)
    self.observation_spaces = {
        aname: Box(low=np.zeros(num_state_bits), high=np.ones(num_state_bits))
        for aname in self.possible_agents
    }

    self.action_spaces = {
        self.possible_agents[idx]: Discrete(mdp.list_num_actions[idx])
        for idx in range(len(self.possible_agents))
    }

  def conv_state_to_obs(self, state_idx, joint_prev_action=None):

    def get_is_neighboring_and_rel_pos(pos_me, pos_target):
      x = pos_target[0] - pos_me[0]
      y = pos_target[1] - pos_me[1]

      rel_pos = (x, y)
      is_neighboring = (x in [-1, 0, 1] and y in [-1, 0, 1])

      return is_neighboring, rel_pos

    def get_box_idxs(pos, box_states):
      is_goal_neigh, _ = get_is_neighboring_and_rel_pos(pos, self.mmdp.goals[0])

      list_box_idxs = []
      for idx in range(len(self.mmdp.boxes)):
        is_box_neigh, _ = get_is_neighboring_and_rel_pos(
            pos, self.mmdp.boxes[idx])

        bst = box_states[idx][0]

        box_aware = None
        if bst == BoxState.WithBoth:
          box_aware = BoxState.WithBoth
        elif is_box_neigh and bst == BoxState.Original:
          box_aware = BoxState.Original
        # elif is_box_neigh:  # near box but not there nor with us --> goal
        #   box_aware = BoxState.OnGoalLoc
        elif bst == BoxState.OnGoalLoc and is_goal_neigh:
          box_aware = BoxState.OnGoalLoc

        list_box_idxs.append(self.dict_boxstate_to_idx[box_aware])

      return tuple(list_box_idxs)

    def get_obs(my_name, fr_name, my_pos, fr_pos, b_states, actions):
      # my_x, my_y, my_a, fr_o, fr_x, fr_y, fr_a, box1, box2, box3
      def one_hot(n, dim):
        np_oh = np.zeros(dim)
        np_oh[n] = 1
        return np_oh

      my_x = one_hot(my_pos[0], self.mmdp.x_grid)
      my_y = one_hot(my_pos[1], self.mmdp.y_grid)
      my_a = one_hot(actions[my_name], self.action_spaces[my_name].n)

      is_nei, rel_pos = get_is_neighboring_and_rel_pos(my_pos, fr_pos)
      fr_o = np.array([int(is_nei)])
      if is_nei:
        fr_x = one_hot(1 + rel_pos[0], 3)
        fr_y = one_hot(1 + rel_pos[1], 3)
        fr_a = one_hot(actions[fr_name], self.action_spaces[fr_name].n)
      else:
        fr_x = np.zeros(3)
        fr_y = np.zeros(3)
        fr_a = np.zeros(self.action_spaces[fr_name].n)

      list_obs_factors = [my_x, my_y, my_a, fr_o, fr_x, fr_y, fr_a]

      box_idxs = get_box_idxs(my_pos, b_states)
      for bidx in box_idxs:
        list_obs_factors.append(one_hot(bidx, len(self.possible_box_state)))

      return np.hstack(list_obs_factors)

    len_s_space = len(self.mmdp.dict_factored_statespace)
    state_vec = self.mmdp.conv_idx_to_state(state_idx)

    a1pos = self.mmdp.pos1_space.idx_to_state[state_vec[0]]
    a2pos = self.mmdp.pos2_space.idx_to_state[state_vec[1]]

    box_states = []
    for idx in range(2, len_s_space):
      box_sidx = state_vec[idx]
      box_state = self.mmdp.dict_factored_statespace[idx].idx_to_state[box_sidx]
      box_states.append(box_state)

    a1name = self.possible_agents[0]
    a2name = self.possible_agents[1]

    dict_obs = {}
    dict_obs[a1name] = get_obs(a1name, a2name, a1pos, a2pos, box_states,
                               joint_prev_action)
    dict_obs[a2name] = get_obs(a2name, a1name, a2pos, a1pos, box_states,
                               joint_prev_action)

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

    actions = {aname: 0 for aname in self.agents}
    observations = self.conv_state_to_obs(self.cur_state, actions)
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

    self.cur_state = self.mmdp.transition(self.cur_state, aidx)
    obs = self.conv_state_to_obs(self.cur_state, actions)

    if self.mmdp.is_terminal(self.cur_state):
      dones = {agent: True for agent in self.agents}
      truncs = {agent: False for agent in self.agents}
    else:
      dones = {agent: False for agent in self.agents}
      truncs = {agent: trunc for agent in self.agents}

    rewards = {agent: -1 for agent in self.agents}

    return obs, rewards, dones, truncs, infos


class PO_Movers_AIAgent(BoxPushAIAgent_PO_Team):

  def __init__(self,
               init_tup_states,
               policy_model: CachedPolicyInterface,
               possible_box_state,
               agent_idx: int = 0) -> None:
    super().__init__(init_tup_states, policy_model, True, agent_idx)
    self.possible_box_state = possible_box_state
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
    my_x = np.where(obs[0:7] == 1)[0][0]
    my_y = np.where(obs[7:14] == 1)[0][0]
    my_a = np.where(obs[14:20] == 1)[0][0]
    fr_o = bool(obs[20])
    box1 = np.where(obs[33:37] == 1)[0][0]
    box2 = np.where(obs[37:41] == 1)[0][0]
    box3 = np.where(obs[41:45] == 1)[0][0]
    obs_boxstates = [box1, box2, box3]

    mdp = self.agent_model.get_reference_mdp()
    num_drops = len(mdp.drops)
    prev_box_states, prev_a1_pos, prev_a2_pos = self.assumed_tup_states

    def max_dist(my_pos, fr_pos):
      return max(abs(my_pos[0] - fr_pos[0]), abs(my_pos[1] - fr_pos[1]))

    def assumed_state(p_fr_pos):
      my_pos = (my_x, my_y)
      fr_pos = None
      if fr_o:
        fr_x = np.where(obs[21:24] == 1)[0][0] - 1
        fr_y = np.where(obs[24:27] == 1)[0][0] - 1
        fr_a = np.where(obs[27:33] == 1)[0][0]
        fr_pos = (my_x + fr_x, my_y + fr_y)
      else:
        # set fr_pos to some random position
        for x in range(mdp.x_grid):
          for y in range(mdp.y_grid):
            crd = (x, y)
            if crd[0] < 0 or crd[0] >= mdp.x_grid:
              continue
            if crd[1] < 0 or crd[1] >= mdp.y_grid:
              continue
            if crd in mdp.goals:
              continue
            if crd in mdp.walls:
              continue
            dist_tmp = max_dist(my_pos, crd)
            if dist_tmp > 1:
              fr_pos = crd
              break

      if fr_pos is None:
        print(my_pos, my_a)
        print(p_fr_pos)
        raise ValueError("fr_pos is None")

      assumed_box_state = list(prev_box_states)
      for idx, bidx in enumerate(obs_boxstates):
        bst = self.possible_box_state[bidx]
        if bst is not None:  # observed
          assumed_box_state[idx] = conv_box_state_2_idx((bst, 0), num_drops)
        # unobserved but near box orig
        elif max_dist(my_pos, mdp.boxes[idx]) <= 1:
          assumed_box_state[idx] = conv_box_state_2_idx((BoxState.OnGoalLoc, 0),
                                                        num_drops)
      return tuple(assumed_box_state), my_pos, fr_pos

    if self.agent_idx == 0:
      box_states, a1_pos, a2_pos = assumed_state(prev_a2_pos)
    else:
      box_states, a2_pos, a1_pos = assumed_state(prev_a1_pos)

    return box_states, a1_pos, a2_pos

  def observed_actions(self, tup_actions, obs) -> tuple:
    my_a = np.where(obs[14:20] == 1)[0][0]
    fr_o = bool(obs[20])

    observed_actions = [None, None]
    my_idx, fr_idx = (0, 1) if self.agent_idx == 0 else (1, 0)

    observed_actions[my_idx] = AGENT_ACTIONSPACE.idx_to_action[my_a]
    if fr_o:
      fr_a = np.where(obs[27:33] == 1)[0][0]
      observed_actions[fr_idx] = AGENT_ACTIONSPACE.idx_to_action[fr_a]

    return tuple(observed_actions)


def generate_data(save_dir, env_name, n_traj, render):
  expert_trajs = defaultdict(list)
  if env_name == "PO_Movers-v2":
    env = PO_Movers_V2(render_mode="human")
  else:
    raise NotImplementedError()

  TEMPERATURE = 0.3
  GAME_MAP = MAP_MOVERS
  MDP_TASK = MDP_Movers_Task(**GAME_MAP)
  MDP_AGENT = MDP_Movers_Agent(**GAME_MAP)
  POLICY_1 = Policy_Movers(MDP_TASK, MDP_AGENT, TEMPERATURE, 0)
  POLICY_2 = Policy_Movers(MDP_TASK, MDP_AGENT, TEMPERATURE, 1)
  init_states = ([0] * len(GAME_MAP["boxes"]), GAME_MAP["a1_init"],
                 GAME_MAP["a2_init"])

  agents = {
      env.possible_agents[0]:
      PO_Movers_AIAgent(init_states,
                        POLICY_1,
                        env.possible_box_state,
                        agent_idx=0),
      env.possible_agents[1]:
      PO_Movers_AIAgent(init_states,
                        POLICY_2,
                        env.possible_box_state,
                        agent_idx=1)
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

  traj = generate_data(cur_dir, "PO_Movers-v2", 100, False)

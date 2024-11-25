import torch
import numpy as np
from omegaconf import DictConfig
from pettingzoo.utils.env import ParallelEnv
from gymnasium.spaces import Discrete, Box
from .option_gail import OptionGAIL, GAIL
from .option_ppo import OptionPPO, PPO
from ....helper.utils import (conv_input, conv_tuple_input, split_by_size,
                              InterfaceHAgent)


def make_agent(config: DictConfig, env: ParallelEnv, agent_idx, use_option):

  agent_name = env.agents[agent_idx]
  latent_dim = config.dim_c[agent_idx] if use_option else 0

  obs_space = env.observation_space(agent_name)
  if isinstance(obs_space, Discrete):
    obs_dim = obs_space.n
    discrete_obs = True
  else:
    obs_dim = obs_space.shape[0]
    discrete_obs = False

  list_aux_dim = []
  list_discrete_aux = []
  list_others_action_dim = []
  list_discrete_others_action = []
  for name in env.agents:
    act_space = env.action_space(name)
    if not (isinstance(act_space, Discrete) or isinstance(act_space, Box)):
      raise RuntimeError(
          "Invalid action space: Only Discrete and Box action spaces supported")

    if isinstance(act_space, Discrete):
      tmp_action_dim = act_space.n
      tmp_discrete_act = True
    else:
      tmp_action_dim = act_space.shape[0]
      tmp_discrete_act = False

    if name == agent_name:
      action_dim = tmp_action_dim
      discrete_act = tmp_discrete_act
    else:
      list_others_action_dim.append(tmp_action_dim)
      list_discrete_others_action.append(tmp_discrete_act)

    if config.use_auxiliary_obs:
      list_aux_dim.append(tmp_action_dim)
      list_discrete_aux.append(tmp_discrete_act)

  agent = MA_OGAIL(config, use_option, obs_dim, action_dim, latent_dim,
                   tuple(list_aux_dim), discrete_obs, discrete_act,
                   tuple(list_discrete_aux), tuple(list_others_action_dim),
                   tuple(list_discrete_others_action))
  return agent


class MA_OGAIL(InterfaceHAgent):

  def __init__(self,
               config: DictConfig,
               use_option,
               obs_dim,
               action_dim,
               lat_dim,
               tup_aux_dim,
               discrete_obs,
               discrete_act,
               tup_discrete_aux,
               tup_oth_dim=None,
               tup_discrete_oth=None):
    self.obs_dim = obs_dim
    self.action_dim = action_dim
    self.lat_dim = lat_dim
    self.tup_aux_dim = tup_aux_dim
    self.discrete_obs = discrete_obs
    self.discrete_act = discrete_act
    self.tup_discrete_aux = tup_discrete_aux
    self.batch_size = config.mini_batch_size

    self.device = torch.device(config.device)
    self.PREV_LATENT = lat_dim

    if config.use_auxiliary_obs and len(tup_aux_dim) > 0:
      prev_aux = []
      aux_split_size = []
      for idx in range(len(tup_aux_dim)):
        if tup_discrete_aux[idx]:
          prev_aux.append(np.array([tup_aux_dim[idx]], dtype=np.float32))
          aux_split_size.append(1)
        else:
          prev_aux.append(np.zeros(tup_aux_dim[idx], dtype=np.float32))
          aux_split_size.append(tup_aux_dim[idx])

      self.PREV_AUX = np.concatenate(prev_aux)
      self.AUX_SPLIT_SIZE = aux_split_size
    else:
      self.tup_aux_dim = tup_aux_dim = ()
      self.tup_discrete_aux = tup_discrete_aux = ()
      self.PREV_AUX = float("nan")  # dummy
      self.AUX_SPLIT_SIZE = []

    self.use_option = use_option

    if self.use_option:
      # not use others' actions for option-gail
      self.tup_oth_dim = ()
      self.tup_discrete_oth = ()
      self.OTHERS_ACTION_SPLIT_SIZE = []
      self.gail = OptionGAIL(config,
                             dim_disc_obs=obs_dim + sum(tup_aux_dim),
                             dim_low_obs=obs_dim,
                             dim_high_obs=obs_dim + sum(tup_aux_dim),
                             dim_a=action_dim,
                             dim_c=lat_dim,
                             discrete_a=discrete_act)
      self.ppo = OptionPPO(config,
                           self.gail.policy,
                           critic_dim=obs_dim + sum(tup_aux_dim))
    else:
      oth_act_split_size = []
      for idx in range(len(tup_oth_dim)):
        if tup_discrete_oth[idx]:
          oth_act_split_size.append(1)
        else:
          oth_act_split_size.append(tup_oth_dim[idx])

      self.tup_oth_dim = tup_oth_dim
      self.tup_discrete_oth = tup_discrete_oth
      self.OTHERS_ACTION_SPLIT_SIZE = oth_act_split_size

      self.gail = GAIL(config,
                       dim_s=obs_dim + sum(tup_aux_dim),
                       dim_a=action_dim,
                       discrete_a=discrete_act)
      # following MAGAIL paper, set critic to depend on other agent actions
      self.ppo = PPO(config,
                     self.gail.policy,
                     critic_dim=obs_dim + sum(tup_aux_dim) + sum(tup_oth_dim))

  def gail_update(self, policy_trajs, expert_trajs, n_step=10):
    # convert sample format
    if self.use_option:
      MERGE_KEYS = ["states", "actions", "prev_latents", "latents", "prev_auxs"]
    else:
      MERGE_KEYS = ["states", "actions", "prev_auxs"]

    num_policy_trajs = len(policy_trajs["states"])
    policy_data = {}
    for key in MERGE_KEYS:
      list_tmp = []
      for i_e in range(num_policy_trajs):
        len_epi = policy_trajs["lengths"][i_e]
        list_tmp.append(np.array(policy_trajs[key][i_e]).reshape(len_epi, -1))
      policy_data[key] = torch.as_tensor(np.vstack(list_tmp),
                                         dtype=torch.float,
                                         device=self.device)

    num_expert_trajs = len(expert_trajs["states"])
    expert_data = {}
    for key in MERGE_KEYS:
      list_tmp = []
      for i_e in range(num_expert_trajs):
        len_epi = expert_trajs["lengths"][i_e]
        list_tmp.append(np.array(expert_trajs[key][i_e]).reshape(len_epi, -1))
      expert_data[key] = torch.as_tensor(np.vstack(list_tmp),
                                         dtype=torch.float,
                                         device=self.device)

    # update
    for _ in range(n_step):
      n_p_samples = policy_data["states"].size(0)
      n_e_samples = expert_data["states"].size(0)
      inds = torch.randperm(n_p_samples, device=self.device)
      for ind_p in inds.split(self.batch_size):
        batch_pol_state = policy_data["states"][ind_p]
        batch_pol_action = policy_data["actions"][ind_p]
        batch_pol_pre_aux = policy_data["prev_auxs"][ind_p]

        bat_sz = ind_p.size(0)
        ind_e = torch.randperm(n_e_samples, device=self.device)[:bat_sz]
        batch_exp_state = expert_data["states"][ind_e]
        batch_exp_action = expert_data["actions"][ind_e]
        batch_exp_pre_aux = expert_data["prev_auxs"][ind_e]

        batch_state = torch.cat((batch_pol_state, batch_exp_state), dim=0)
        batch_action = torch.cat((batch_pol_action, batch_exp_action), dim=0)
        batch_pre_aux = torch.cat((batch_pol_pre_aux, batch_exp_pre_aux), dim=0)

        batch_is_policy = torch.cat(
            (torch.ones(bat_sz, 1, dtype=torch.float32, device=self.device),
             torch.zeros(bat_sz, 1, dtype=torch.float32, device=self.device)),
            dim=0)

        tup_prev_aux = split_by_size(batch_pre_aux, self.AUX_SPLIT_SIZE,
                                     self.device)
        batch_disc_obs = conv_tuple_input(
            (batch_state, *tup_prev_aux),
            (self.discrete_obs, *self.tup_discrete_aux),
            (self.obs_dim, *self.tup_aux_dim), self.device)
        batch_action = conv_input(batch_action, self.discrete_act,
                                  self.action_dim, self.device)

        if self.use_option:
          batch_pol_pre_lat = policy_data["prev_latents"][ind_p]
          batch_pol_latent = policy_data["latents"][ind_p]

          batch_exp_pre_lat = expert_data["prev_latents"][ind_e]
          batch_exp_latent = expert_data["latents"][ind_e]

          batch_pre_lat = torch.cat((batch_pol_pre_lat, batch_exp_pre_lat),
                                    dim=0)
          batch_latent = torch.cat((batch_pol_latent, batch_exp_latent), dim=0)

          batch_pre_lat = conv_input(batch_pre_lat, False, 1, self.device)
          batch_latent = conv_input(batch_latent, False, 1, self.device)

          for _ in range(3):
            self.gail.step(batch_disc_obs, batch_pre_lat, batch_action,
                           batch_latent, batch_is_policy)
        else:
          for _ in range(3):
            self.gail.step(batch_disc_obs, batch_action, batch_is_policy)

  def ppo_update(self, policy_trajs, n_step=10):
    num_trajs = len(policy_trajs["states"])
    NEW_KEYS_OPTION = [
        "returns", "adv_hi", "adv_lo", "v_val_hi", "v_val_lo", "pc", "log_p_hi",
        "log_p_lo"
    ]
    NEW_KEYS_ORIGINAL = ["returns", "adv", "v_val", "log_probs"]
    for i_e in range(num_trajs):
      obs = policy_trajs["states"][i_e]
      actions = policy_trajs["actions"][i_e]
      reward = torch.as_tensor(policy_trajs["rewards"][i_e],
                               dtype=torch.float32,
                               device=self.device)
      prev_aux = policy_trajs["prev_auxs"][i_e]

      tup_prev_aux = split_by_size(prev_aux, self.AUX_SPLIT_SIZE, self.device)
      high_obs = conv_tuple_input(
          (obs, *tup_prev_aux), (self.discrete_obs, *self.tup_discrete_aux),
          (self.obs_dim, *self.tup_aux_dim), self.device)

      if self.discrete_act:
        actions = conv_input(actions, False, 1, self.device)
      else:
        actions = conv_input(actions, self.discrete_act, self.action_dim,
                             self.device)

      if self.use_option:
        prev_lat = policy_trajs["prev_latents"][i_e]
        latent = policy_trajs["latents"][i_e]
        low_obs = conv_input(obs, self.discrete_obs, self.obs_dim, self.device)
        prev_lat = conv_input(prev_lat, False, 1, self.device)
        latent = conv_input(latent, False, 1, self.device)
        vec_interm = self.ppo.calc_adv_each_episode(low_obs, actions, high_obs,
                                                    prev_lat, latent, reward)
        for idx, item in enumerate(vec_interm):
          policy_trajs[NEW_KEYS_OPTION[idx]].append(item)
      else:
        others_actions = policy_trajs["others_actions"][i_e]
        tup_others_actions = split_by_size(others_actions,
                                           self.OTHERS_ACTION_SPLIT_SIZE,
                                           self.device)
        critic_obs = conv_tuple_input(
            (obs, *tup_prev_aux, *tup_others_actions),
            (self.discrete_obs, *self.tup_discrete_aux, *self.tup_discrete_oth),
            (self.obs_dim, *self.tup_aux_dim, *self.tup_oth_dim), self.device)

        vec_interm = self.ppo.calc_adv_each_episode(high_obs, actions,
                                                    critic_obs, reward)
        for idx, item in enumerate(vec_interm):
          policy_trajs[NEW_KEYS_ORIGINAL[idx]].append(item)

    # merged data
    policy_data = {}
    for key in policy_trajs.keys():
      if key in ["lengths", "dones"]:
        continue

      list_tmp = []
      for i_e in range(num_trajs):
        len_epi = policy_trajs["lengths"][i_e]
        tmp_item = policy_trajs[key][i_e]
        if not isinstance(tmp_item, torch.Tensor):
          tmp_item = torch.as_tensor(np.array(tmp_item),
                                     dtype=torch.float32,
                                     device=self.device)
        list_tmp.append(tmp_item.reshape(len_epi, -1))

      policy_data[key] = torch.cat(list_tmp, dim=0)

    # update
    for _ in range(n_step):
      inds = torch.randperm(policy_data["states"].size(0), device=self.device)
      for ind_b in inds.split(self.batch_size):
        obs = policy_data["states"][ind_b]
        actions = policy_data["actions"][ind_b]
        prev_aux = policy_data["prev_auxs"][ind_b]
        returns = policy_data["returns"][ind_b]

        tup_prev_aux = split_by_size(prev_aux, self.AUX_SPLIT_SIZE, self.device)
        high_obs = conv_tuple_input(
            (obs, *tup_prev_aux), (self.discrete_obs, *self.tup_discrete_aux),
            (self.obs_dim, *self.tup_aux_dim), self.device)

        if self.use_option:
          latent = policy_data["latents"][ind_b]
          prev_lat = policy_data["prev_latents"][ind_b]
          adv_hi = policy_data["adv_hi"][ind_b]
          adv_lo = policy_data["adv_lo"][ind_b]
          v_val_hi = policy_data["v_val_hi"][ind_b]
          v_val_lo = policy_data["v_val_lo"][ind_b]
          fixed_pc = policy_data["pc"][ind_b]
          fixed_logp_hi = policy_data["log_p_hi"][ind_b]
          fixed_logp_lo = policy_data["log_p_lo"][ind_b]

          low_obs = conv_input(obs, self.discrete_obs, self.obs_dim,
                               self.device)

          self.ppo.step(low_obs, actions, high_obs, prev_lat, latent, returns,
                        adv_hi, adv_lo, v_val_hi, v_val_lo, fixed_pc,
                        fixed_logp_hi, fixed_logp_lo)
        else:
          adv = policy_data["adv"][ind_b]
          v_val = policy_data["v_val"][ind_b]
          fixed_logp = policy_data["log_probs"][ind_b]
          others_actions = policy_data["others_actions"][ind_b]

          tup_others_actions = split_by_size(others_actions,
                                             self.OTHERS_ACTION_SPLIT_SIZE,
                                             self.device)
          critic_obs = conv_tuple_input(
              (obs, *tup_prev_aux, *tup_others_actions),
              (self.discrete_obs, *self.tup_discrete_aux,
               *self.tup_discrete_oth),
              (self.obs_dim, *self.tup_aux_dim, *self.tup_oth_dim), self.device)

          self.ppo.step(high_obs, actions, critic_obs, returns, adv, v_val,
                        fixed_logp)

  def gail_reward(self, obs, prev_lat, prev_aux, action, lat):
    tup_prev_aux = split_by_size(prev_aux, self.AUX_SPLIT_SIZE, self.device)
    disc_obs = conv_tuple_input((obs, *tup_prev_aux),
                                (self.discrete_obs, *self.tup_discrete_aux),
                                (self.obs_dim, *self.tup_aux_dim), self.device)
    action = conv_input(action, self.discrete_act, self.action_dim, self.device)

    with torch.no_grad():
      if self.use_option:
        prev_lat = conv_input(prev_lat, False, 1, self.device)
        lat = conv_input(lat, False, 1, self.device)
        return self.gail.gail_reward(disc_obs, prev_lat, action,
                                     lat).cpu().numpy()
      else:
        return self.gail.gail_reward(disc_obs, action).cpu().numpy()

  def choose_action(self,
                    obs,
                    prev_option,
                    prev_aux,
                    sample=False,
                    avail_actions=None):

    if self.use_option:
      option = self.choose_mental_state(obs, prev_option, prev_aux, sample)
      action = self.choose_policy_action(obs, option, sample, avail_actions)
    else:
      option = self.PREV_LATENT

      tup_prev_aux = split_by_size(prev_aux, self.AUX_SPLIT_SIZE, self.device)
      policy_obs = conv_tuple_input(
          (obs, *tup_prev_aux), (self.discrete_obs, *self.tup_discrete_aux),
          (self.obs_dim, *self.tup_aux_dim), self.device)
      with torch.no_grad():
        action = self.gail.policy.sample_action(policy_obs, not sample,
                                                avail_actions)[0].cpu().numpy()
    return option, action

  def choose_policy_action(self, obs, option, sample=False, avail_actions=None):
    obs = conv_input(obs, self.discrete_obs, self.obs_dim, self.device)
    option = conv_input(option, False, 1, self.device)
    with torch.no_grad():
      return self.gail.policy.sample_action(obs, option, not sample,
                                            avail_actions)[0].cpu().numpy()

  def choose_mental_state(self, obs, prev_option, prev_aux, sample=False):
    tup_prev_aux = split_by_size(prev_aux, self.AUX_SPLIT_SIZE, self.device)
    high_obs = conv_tuple_input((obs, *tup_prev_aux),
                                (self.discrete_obs, *self.tup_discrete_aux),
                                (self.obs_dim, *self.tup_aux_dim), self.device)
    prev_option = conv_input(prev_option, False, 1, self.device)

    with torch.no_grad():
      return self.gail.policy.sample_option(high_obs, prev_option,
                                            not sample)[0].cpu().numpy()[0]

  def save(self, path):
    torch.save(self.gail.discriminator.state_dict(), path + "_disc")
    torch.save(self.gail.policy.state_dict(), path + "_policy")
    torch.save(self.ppo.critic.state_dict(), path + "_critic")

  def load(self, path):
    print('Loading models from {}'.format(path))
    self.gail.discriminator.load_state_dict(
        torch.load(path + "_disc", self.device))
    self.gail.policy.load_state_dict(torch.load(path + "_policy", self.device))
    self.ppo.critic.load_state_dict(torch.load(path + "_critic", self.device))

  def infer_mental_states(self, obs, action, prev_aux):
    '''
    return: options with the length of len_demo
    '''
    # c_array, log_prob_traj = self.policy.viterbi_path(s_array, a_array)
    # return c_array[1:].cpu().numpy(), log_prob_traj.cpu().numpy()

    len_demo = len(obs)

    batch_prev_aux_split = split_by_size(prev_aux, self.AUX_SPLIT_SIZE,
                                         self.device)
    low_obs = conv_tuple_input((obs, *batch_prev_aux_split),
                               (self.discrete_obs, *self.tup_discrete_aux),
                               (self.obs_dim, *self.tup_aux_dim), self.device)
    high_obs = conv_input(obs, self.discrete_obs, self.obs_dim, self.device)
    if self.discrete_act:
      action = conv_input(action, False, 1, self.device)
    else:
      action = conv_input(action, self.discrete_act, self.action_dim,
                          self.device)

    with torch.no_grad():
      log_pis = self.gail.policy.log_prob_action(low_obs, None, action).view(
          -1, 1, self.lat_dim)  # demo_len x 1 x ct
      log_trs = self.gail.policy.log_trans(high_obs,
                                           None)  # demo_len x (ct_1+1) x ct
      log_prob = log_trs[:, :-1] + log_pis
      log_prob0 = log_trs[0, -1] + log_pis[0, 0]
      # forward
      max_path = torch.empty(len_demo,
                             self.lat_dim,
                             dtype=torch.long,
                             device=self.device)
      accumulate_logp = log_prob0
      max_path[0] = self.lat_dim
      for i in range(1, len_demo):
        accumulate_logp, max_path[i, :] = (accumulate_logp.unsqueeze(dim=-1) +
                                           log_prob[i]).max(dim=-2)
      # backward
      c_array = torch.zeros(len_demo + 1,
                            1,
                            dtype=torch.long,
                            device=self.device)
      log_prob_traj, c_array[-1] = accumulate_logp.max(dim=-1)
      for i in range(len_demo, 0, -1):
        c_array[i - 1] = max_path[i - 1][c_array[i]]
    return (c_array[1:].detach().cpu().numpy(),
            log_prob_traj.detach().cpu().numpy())

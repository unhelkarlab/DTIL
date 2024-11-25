import torch
import numpy as np
from .nn_models import (SimpleOptionQNetwork, DoubleOptionQCritic,
                        SingleOptionQCritic, DiagGaussianOptionActor)
from .option_iql import IQLOptionSAC, IQLOptionSoftQ
from omegaconf import DictConfig
from ..helper.utils import split_by_size


def get_tx_pi_config(config: DictConfig):
  tx_prefix = "mahil_tx_"
  config_tx = DictConfig({})
  for key in config:
    if key[:len(tx_prefix)] == tx_prefix:
      config_tx[key[len(tx_prefix):]] = config[key]

  pi_prefix = "mahil_pi_"
  config_pi = DictConfig({})
  for key in config:
    if key[:len(pi_prefix)] == pi_prefix:
      config_pi[key[len(pi_prefix):]] = config[key]

  config_pi["gamma"] = config_tx["gamma"] = config.gamma
  config_pi["device"] = config_tx["device"] = config.device

  return config_tx, config_pi


class MAHIL:

  def __init__(self, config: DictConfig, obs_dim, action_dim, lat_dim,
               tup_aux_dim, discrete_obs, discrete_act, tup_discrete_aux):
    self.obs_dim = obs_dim
    self.action_dim = action_dim
    self.lat_dim = lat_dim
    self.discrete_obs = discrete_obs
    self.discrete_act = discrete_act
    self.tup_discrete_aux = tup_discrete_aux

    self.update_strategy = config.mahil_update_strategy
    self.update_tx_after_pi = config.mahil_tx_after_pi
    self.alter_update_n_pi_tx = config.mahil_alter_update_n_pi_tx
    self.order_update_pi_ratio = config.mahil_order_update_pi_ratio

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
      tup_aux_dim = ()
      tup_discrete_aux = ()
      self.PREV_AUX = float("nan")  # dummy
      self.AUX_SPLIT_SIZE = []

    self.internal_step = 0
    self.pi_update_count = 0
    self.tx_update_count = 0

    config_tx, config_pi = get_tx_pi_config(config)

    tup_tx_obs_dim = (obs_dim, *tup_aux_dim)
    tup_tx_discrete_obs = (discrete_obs, *tup_discrete_aux)
    self.tx_agent = IQLOptionSoftQ(config_tx, tup_tx_obs_dim, lat_dim,
                                   lat_dim + 1, tup_tx_discrete_obs,
                                   SimpleOptionQNetwork, self._get_tx_iq_vars)

    tup_pi_obs_dim = (obs_dim, )
    tup_pi_discrete_obs = (discrete_obs, )
    if discrete_act:
      self.pi_agent = IQLOptionSoftQ(config_pi, tup_pi_obs_dim, action_dim,
                                     lat_dim, tup_pi_discrete_obs,
                                     SimpleOptionQNetwork, self._get_pi_iq_vars)
    else:
      if config.mahil_pi_single_critic:
        critic_base = SingleOptionQCritic
      else:
        critic_base = DoubleOptionQCritic
      actor = DiagGaussianOptionActor(
          sum(tup_pi_obs_dim), action_dim, lat_dim, config_pi.hidden_policy,
          config_pi.activation, config_pi.log_std_bounds,
          config_pi.bounded_actor, config_pi.use_nn_logstd,
          config_pi.clamp_action_logstd)
      self.pi_agent = IQLOptionSAC(config_pi, tup_pi_obs_dim, action_dim,
                                   lat_dim, tup_pi_discrete_obs, critic_base,
                                   actor, self._get_pi_iq_vars)

    self.train()

  def train(self, training=True):
    self.training = training
    self.tx_agent.train(training)
    self.pi_agent.train(training)

  def reset_optimizers(self):
    self.tx_agent.reset_optimizers()
    self.pi_agent.reset_optimizers()

  def _get_tx_iq_vars(self, batch):
    batch_prev_aux_split = split_by_size(batch['prev_auxs'],
                                         self.AUX_SPLIT_SIZE, self.device)
    batch_aux_split = split_by_size(batch['auxs'], self.AUX_SPLIT_SIZE,
                                    self.device)

    tup_obs = (batch['states'], *batch_prev_aux_split)
    vec_v_args = (tup_obs, batch['prev_latents'])

    tup_next_obs = (batch['next_states'], *batch_aux_split)
    latent = batch['latents']
    vec_next_v_args = (tup_next_obs, latent)
    vec_actions = (latent, )
    return vec_v_args, vec_next_v_args, vec_actions

  def _get_pi_iq_vars(self, batch):
    vec_v_args = ((batch['states'], ), batch['latents'])
    vec_next_v_args = ((batch['next_states'], ), batch['next_latents'])
    vec_actions = (batch['actions'], )
    return vec_v_args, vec_next_v_args, vec_actions

  def pi_update(self, policy_batch, expert_batch, logger, step):
    if self.discrete_act:
      pi_use_target, pi_soft_update = False, False
    else:
      pi_use_target, pi_soft_update = True, True

    pi_loss = self.pi_agent.iq_update(policy_batch, expert_batch, logger,
                                      self.pi_update_count, pi_use_target,
                                      pi_soft_update, self.pi_agent.method_loss,
                                      self.pi_agent.method_regularize,
                                      self.pi_agent.method_div)
    self.pi_update_count += 1
    return pi_loss

  def tx_update(self, policy_batch, expert_batch, logger, step):
    if self.lat_dim == 1:
      return {}

    TX_USE_TARGET, TX_DO_SOFT_UPDATE = False, False
    tx_loss = self.tx_agent.iq_update(policy_batch, expert_batch, logger,
                                      self.tx_update_count, TX_USE_TARGET,
                                      TX_DO_SOFT_UPDATE,
                                      self.tx_agent.method_loss,
                                      self.tx_agent.method_regularize,
                                      self.tx_agent.method_div)
    self.tx_update_count += 1
    return tx_loss

  def mahil_update(self, policy_batch, expert_batch, num_updates_per_cycle,
                   logger, step):
    # update pi first and then tx
    ALWAYS_UPDATE_BOTH = 1
    UPDATE_IN_ORDER = 2
    UPDATE_ALTERNATIVELY = 3

    if self.internal_step >= num_updates_per_cycle:
      self.internal_step = 0

    self.internal_step += 1

    num_pi_update, num_tx_update = self.alter_update_n_pi_tx

    if self.update_tx_after_pi:
      fn_update_1, fn_update_2 = self.tx_update, self.pi_update
      ratio_1st = (1 - self.order_update_pi_ratio)
      num_1st = num_tx_update
    else:
      fn_update_1, fn_update_2 = self.pi_update, self.tx_update
      ratio_1st = self.order_update_pi_ratio
      num_1st = num_pi_update

    loss_1, loss_2 = {}, {}
    if self.update_strategy == ALWAYS_UPDATE_BOTH:
      loss_1 = fn_update_1(policy_batch, expert_batch, logger, step)
      loss_2 = fn_update_2(policy_batch, expert_batch, logger, step)
    elif self.update_strategy == UPDATE_IN_ORDER:
      if self.internal_step < ratio_1st * num_updates_per_cycle:
        loss_1 = fn_update_1(policy_batch, expert_batch, logger, step)
      else:
        loss_2 = fn_update_2(policy_batch, expert_batch, logger, step)
    elif self.update_strategy == UPDATE_ALTERNATIVELY:
      alternating_step = self.internal_step % (num_pi_update + num_tx_update)
      if alternating_step < num_1st:
        loss_1 = fn_update_1(policy_batch, expert_batch, logger, step)
      else:
        loss_2 = fn_update_2(policy_batch, expert_batch, logger, step)
    else:
      raise NotImplementedError

    return (loss_1, loss_2) if self.update_tx_after_pi else (loss_2, loss_1)

  def mahil_offline_update(self, expert_batch, logger, step):
    TX_USE_TARGET, TX_DO_SOFT_UPDATE = False, False

    if self.discrete_act:
      pi_use_target, pi_soft_update = False, False
    else:
      pi_use_target, pi_soft_update = True, True
    if self.lat_dim == 1:
      loss_1 = {}
    else:
      loss_1 = self.tx_agent.iq_offline_update(expert_batch, logger, step,
                                               TX_USE_TARGET, TX_DO_SOFT_UPDATE,
                                               self.tx_agent.method_regularize,
                                               self.tx_agent.method_div)
    loss_2 = self.pi_agent.iq_offline_update(expert_batch, logger, step,
                                             pi_use_target, pi_soft_update,
                                             self.pi_agent.method_regularize,
                                             self.pi_agent.method_div)
    return loss_1, loss_2

  def choose_action(self,
                    obs,
                    prev_option,
                    prev_aux,
                    sample=False,
                    avail_actions=None):
    'for compatibility with OptionIQL evaluate function'

    option = self.choose_mental_state(obs, prev_option, prev_aux, sample)
    action = self.choose_policy_action(obs,
                                       option,
                                       sample,
                                       avail_actions=avail_actions)
    return option, action

  def choose_policy_action(self, obs, option, sample=False, avail_actions=None):
    if self.discrete_act:
      return self.pi_agent.choose_action((obs, ), option, sample, avail_actions)
    else:
      return self.pi_agent.choose_action((obs, ), option, sample)

  def choose_mental_state(self, obs, prev_option, prev_aux, sample=False):
    if self.lat_dim == 1:
      return 0

    batch_prev_aux_split = split_by_size(prev_aux, self.AUX_SPLIT_SIZE,
                                         self.device)
    tup_obs = (obs, ) + batch_prev_aux_split
    return self.tx_agent.choose_action(tup_obs, prev_option, sample)

  def save(self, path):
    self.tx_agent.save(path, "_tx")
    self.pi_agent.save(path, "_pi")

  def load(self, path):
    self.tx_agent.load(path + "_tx")
    self.pi_agent.load(path + "_pi")

  def infer_mental_states(self, obs, action, prev_aux):
    '''
    return: options with the length of len_demo
    '''
    len_demo = len(obs)

    if self.lat_dim == 1:
      return np.zeros((len_demo, 1)), 0.0

    batch_prev_aux_split = split_by_size(prev_aux, self.AUX_SPLIT_SIZE,
                                         self.device)
    tup_tx_obs = (obs, *batch_prev_aux_split)

    with torch.no_grad():
      log_pis = self.pi_agent.log_probs(
          (obs, ), action).view(-1, 1, self.lat_dim)  # len_demo x 1 x ct
      log_trs = self.tx_agent.log_probs(tup_tx_obs,
                                        None)  # len_demo x (ct_1+1) x ct
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

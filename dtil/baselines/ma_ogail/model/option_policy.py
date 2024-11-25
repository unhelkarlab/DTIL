import math
import torch
import torch.nn.functional as F
from ..utils.model_util import (make_module, make_module_list, make_activation)
from omegaconf import DictConfig

# this policy uses one-step option, the initial option is fixed as o=dim_c


class Policy(torch.nn.Module):

  def __init__(self, config: DictConfig, dim_s, dim_a, discrete_a=False):
    super(Policy, self).__init__()
    self.dim_a = dim_a
    self.dim_s = dim_s
    self.discrete_a = discrete_a
    self.device = torch.device(config.device)
    self.log_clamp = config.log_std_bounds
    activation = make_activation(config.activation)
    n_hidden_pi = config.hidden_policy

    self.policy = make_module(self.dim_s, self.dim_a, n_hidden_pi, activation)
    self.a_log_std = torch.nn.Parameter(
        torch.empty(1, self.dim_a, dtype=torch.float32).fill_(0.))

    self.to(self.device)

  def a_mean_logstd(self, s):
    y = self.policy(s)
    mean, logstd = y, self.a_log_std.expand_as(y)
    return mean.clamp(-10, 10), logstd.clamp(self.log_clamp[0],
                                             self.log_clamp[1])

  def a_log_softmax(self, low_obs):
    logits = self.policy(low_obs)
    log_pas = logits.log_softmax(dim=-1)

    return log_pas

  def log_prob_action(self, s, a):
    if self.discrete_a:
      log_pas = self.a_log_softmax(s)
      return log_pas.gather(dim=-1, index=a.long())
    else:
      mean, logstd = self.a_mean_logstd(s)
      return (-((a - mean)**2) / (2 * (logstd * 2).exp()) - logstd -
              math.log(math.sqrt(2 * math.pi))).sum(dim=-1, keepdim=True)

  def sample_action(self, s, fixed=False, avail_actions=None):
    if self.discrete_a:
      log_pas = self.a_log_softmax(s)
      if avail_actions is not None:
        log_pas[avail_actions.reshape(log_pas.shape) == 0] = -1e10
      # if fixed:
      #   return log_pas.argmax(dim=-1, keepdim=True)[0]
      # else:
      return F.gumbel_softmax(log_pas, hard=False).multinomial(1).long()[0]
    else:
      action_mean, action_log_std = self.a_mean_logstd(s)
      if fixed:
        action = action_mean
      else:
        eps = torch.empty_like(action_mean).normal_()
        action = action_mean + action_log_std.exp() * eps
      return action

  def policy_log_prob_entropy(self, s, a):
    if self.discrete_a:
      log_prob = self.log_prob_action(s, a)
      entropy = -log_prob.mean()
    else:
      mean, logstd = self.a_mean_logstd(s)
      log_prob = (-(a - mean).square() / (2 * (logstd * 2).exp()) - logstd -
                  0.5 * math.log(2 * math.pi)).sum(dim=-1, keepdim=True)
      entropy = (0.5 + 0.5 * math.log(2 * math.pi) + logstd).sum(dim=-1,
                                                                 keepdim=True)
    return log_prob, entropy

  def get_param(self, low_policy=True):
    if not low_policy:
      print("WARNING >>>> policy do not have high policy params, "
            "returning low policy params instead")
    return list(self.parameters())


class OptionPolicy(torch.nn.Module):

  def __init__(self,
               config: DictConfig,
               dim_low_obs,
               dim_high_obs,
               dim_a,
               dim_c,
               discrete_a=False):
    super(OptionPolicy, self).__init__()
    self.dim_a = dim_a
    self.dim_c = dim_c
    self.discrete_a = discrete_a
    self.device = torch.device(config.device)
    self.log_clamp = config.log_std_bounds
    self.is_shared = config.shared_policy
    activation = make_activation(config.activation)
    n_hidden_pi = config.hidden_policy
    n_hidden_opt = config.hidden_option

    if self.is_shared:
      # output prediction p(ct| st, ct-1) with shape (N x ct-1 x ct)
      self.option_policy = make_module(dim_high_obs,
                                       (self.dim_c + 1) * self.dim_c,
                                       n_hidden_opt, activation)
      self.policy = make_module(dim_low_obs, self.dim_c * self.dim_a,
                                n_hidden_pi, activation)

      self.a_log_std = torch.nn.Parameter(
          torch.empty(1, self.dim_a, dtype=torch.float32).fill_(0.))
    else:
      self.policy = make_module_list(dim_low_obs, self.dim_a, n_hidden_pi,
                                     self.dim_c, activation)
      self.a_log_std = torch.nn.ParameterList([
          torch.nn.Parameter(
              torch.empty(1, self.dim_a, dtype=torch.float32).fill_(0.))
          for _ in range(self.dim_c)
      ])
      # i-th model output prediction p(ct|st, ct-1=i)
      self.option_policy = make_module_list(dim_high_obs, self.dim_c,
                                            n_hidden_opt, self.dim_c + 1,
                                            activation)

    self.to(self.device)

  def a_mean_logstd(self, low_obs, ct=None):
    # ct: None or long(N x 1)
    # ct: None for all c, return (N x dim_c x dim_a); else return (N x dim_a)
    # s: N x dim_s, c: N x 1, c should always < dim_c
    if self.is_shared:
      mean = self.policy(low_obs).view(-1, self.dim_c, self.dim_a)
      logstd = self.a_log_std.expand_as(mean[:, 0, :])
    else:
      mean = torch.stack([m(low_obs) for m in self.policy], dim=-2)
      logstd = torch.stack([m.expand_as(mean[:, 0, :]) for m in self.a_log_std],
                           dim=-2)
    if ct is not None:
      ind = ct.long().view(-1, 1, 1).expand(-1, 1, self.dim_a)
      mean = mean.gather(dim=-2, index=ind).squeeze(dim=-2)
      logstd = logstd.gather(dim=-2, index=ind).squeeze(dim=-2)
    return mean.clamp(-10, 10), logstd.clamp(self.log_clamp[0],
                                             self.log_clamp[1])

  def switcher(self, high_obs):
    if self.is_shared:
      return self.option_policy(high_obs).view(-1, self.dim_c + 1, self.dim_c)
    else:
      return torch.stack([m(high_obs) for m in self.option_policy], dim=-2)

  def get_param(self, low_policy=True):
    if low_policy:
      if self.is_shared:
        return list(self.policy.parameters()) + [self.a_log_std]
      else:
        return list(self.policy.parameters()) + list(
            self.a_log_std.parameters())
    else:
      return list(self.option_policy.parameters())

  # ===================================================================== #

  def log_trans(self, high_obs, ct_1=None):
    # ct_1: long(N x 1) or None
    # ct_1: None: direct output p(ct|st, ct_1): a (N x ct_1 x ct) array
    #                                where ct is log-normalized
    unnormed_pcs = self.switcher(high_obs)
    log_pcs = unnormed_pcs.log_softmax(dim=-1)
    if ct_1 is None:
      return log_pcs
    else:
      return log_pcs.gather(dim=-2,
                            index=ct_1.long().view(-1, 1, 1).expand(
                                -1, 1, self.dim_c)).squeeze(dim=-2)

  def a_log_softmax(self, low_obs, ct):
    if self.is_shared:
      logits = self.policy(low_obs).view(-1, self.dim_c, self.dim_a)
    else:
      logits = torch.stack([m(low_obs) for m in self.policy], dim=-2)
    log_pas = logits.log_softmax(dim=-1)
    if ct is not None:
      log_pas = log_pas.gather(dim=-2,
                               index=ct.long().view(-1, 1, 1).expand(
                                   -1, 1, self.dim_a)).squeeze(dim=-2)
    return log_pas

  def log_prob_action(self, low_obs, ct, at):
    # if c is None, return (N x dim_c x 1), else return (N x 1)
    if self.discrete_a:
      log_pas = self.a_log_softmax(low_obs, ct)
      at = at.long()
      if ct is None:
        at = at.long().view(-1, 1, 1).expand(-1, self.dim_c, 1)
      return log_pas.gather(dim=-1, index=at)
    else:
      mean, logstd = self.a_mean_logstd(low_obs, ct)
      if ct is None:
        at = at.view(-1, 1, self.dim_a)
      return (-((at - mean).square()) / (2 * (logstd * 2).exp()) - logstd -
              math.log(math.sqrt(2 * math.pi))).sum(dim=-1, keepdim=True)

  def log_prob_option(self, high_obs, ct_1, ct):
    log_tr = self.log_trans(high_obs, ct_1)
    return log_tr.gather(dim=-1, index=ct.long())

  def sample_action(self, low_obs, ct, fixed=False, avail_actions=None):
    if self.discrete_a:
      log_pas = self.a_log_softmax(low_obs, ct)
      if avail_actions is not None:
        log_pas[avail_actions.reshape(log_pas.shape) == 0] = -1e10
      # if fixed:
      #   return log_pas.argmax(dim=-1, keepdim=True)[0]
      # else:
      return F.gumbel_softmax(log_pas, hard=False).multinomial(1).long()[0]
    else:
      action_mean, action_log_std = self.a_mean_logstd(low_obs, ct)
      if fixed:
        action = action_mean
      else:
        eps = torch.empty_like(action_mean).normal_()
        action = action_mean + action_log_std.exp() * eps
      return action

  def sample_option(self, high_obs, ct_1, fixed=False):
    log_tr = self.log_trans(high_obs, ct_1)
    if fixed:
      return log_tr.argmax(dim=-1, keepdim=True)
    else:
      return F.gumbel_softmax(log_tr, hard=False).multinomial(1).long()

  def policy_entropy(self, low_obs, ct):
    if self.discrete_a:
      raise NotImplementedError

    else:
      _, log_std = self.a_mean_logstd(low_obs, ct)
      entropy = 0.5 + 0.5 * math.log(2 * math.pi) + log_std
      return entropy.sum(dim=-1, keepdim=True)

  def option_entropy(self, high_obs, ct_1):
    log_tr = self.log_trans(high_obs, ct_1)
    entropy = -(log_tr * log_tr.exp()).sum(dim=-1, keepdim=True)
    return entropy

  def policy_log_prob_entropy(self, low_obs, ct, at):
    if self.discrete_a:
      log_prob = self.log_prob_action(low_obs, ct, at)
      entropy = -log_prob.mean()
    else:
      mean, logstd = self.a_mean_logstd(low_obs, ct)
      log_prob = (-(at - mean).pow(2) / (2 * (logstd * 2).exp()) - logstd -
                  0.5 * math.log(2 * math.pi)).sum(dim=-1, keepdim=True)
      entropy = (0.5 + 0.5 * math.log(2 * math.pi) + logstd).sum(dim=-1,
                                                                 keepdim=True)
    return log_prob, entropy

  def option_log_prob_entropy(self, high_obs, ct_1, ct):
    # c1 can be dim_c, c2 should always < dim_c
    log_tr = self.log_trans(high_obs, ct_1)
    log_opt = log_tr.gather(dim=-1, index=ct.long())
    entropy = -(log_tr * log_tr.exp()).sum(dim=-1, keepdim=True)
    return log_opt, entropy

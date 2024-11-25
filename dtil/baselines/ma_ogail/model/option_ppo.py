import torch
from .option_critic import OptionCritic, Critic
from .option_policy import OptionPolicy, Policy
from omegaconf import DictConfig


class PPO:
  def __init__(self, config: DictConfig, policy: Policy, critic_dim):
    self.policy = policy
    self.clip_eps = config.clip_eps
    self.lr = config.optimizer_lr_policy
    self.gamma = config.gamma
    self.gae_tau = config.gae_tau
    self.use_gae = config.use_gae
    self.mini_bs = config.mini_batch_size
    self.lambda_entropy = config.lambda_entropy_policy

    self.critic = Critic(config, critic_dim)

    self.reset_optimizer()

  def reset_optimizer(self, lr_mult=1.0):
    self.optim = torch.optim.Adam(self.critic.get_param() +
                                  self.policy.get_param(),
                                  lr=self.lr * lr_mult,
                                  weight_decay=1.e-3,
                                  eps=1e-5)

  def calc_adv_each_episode(self, obs, actions, critic_obs, reward):
    with torch.no_grad():
      v_val = self.critic.get_value(critic_obs).detach()
      advantages = torch.zeros_like(v_val)
      returns = torch.zeros_like(v_val)
      next_value = 0.
      adv = 0.
      ret = 0.

      for i in reversed(range(reward.size(0))):
        ret = reward[i] + self.gamma * ret
        returns[i] = ret

        if not self.use_gae:
          advantages[i] = ret - v_val[i]
        else:
          delta = reward[i] + self.gamma * next_value - v_val[i]
          adv = delta + self.gamma * self.gae_tau * adv
          advantages[i] = adv
          next_value = v_val[i]

      log_probs = self.policy.log_prob_action(obs, actions).detach()

    return returns, advantages, v_val, log_probs

  def step(self, obs, actions, critic_obs, returns, adv, v_val, fixed_logp):
    n_batch = len(returns)

    adv = (adv - adv.mean()) / (adv.std() + 1e-8) if n_batch > 1 else 0.

    logp, entropy = self.policy.policy_log_prob_entropy(obs, actions)
    vpred = self.critic.get_value(critic_obs)

    vpred_clip = v_val + (vpred - v_val).clamp(-self.clip_eps, self.clip_eps)
    vf_loss = torch.max((vpred - returns).square(),
                        (vpred_clip - returns).square()).mean()

    ratio = (logp - fixed_logp).clamp_max(15.).exp()
    pg_loss = -torch.min(
        adv * ratio,
        adv * ratio.clamp(1.0 - self.clip_eps, 1.0 + self.clip_eps)).mean()
    loss = pg_loss + vf_loss * 0.5 - self.lambda_entropy * entropy.mean()
    self.optim.zero_grad()
    loss.backward()
    # after many experiments i find that do not clamp performs the best
    # torch.nn.utils.clip_grad_norm_(self.policy.get_param(), 0.5)
    self.optim.step()


class OptionPPO(torch.nn.Module):
  def __init__(self, config: DictConfig, policy: OptionPolicy, critic_dim):
    super(OptionPPO, self).__init__()
    self.train_policy = config.train_policy
    self.train_option = config.train_option
    self.gamma = config.gamma
    self.gae_tau = config.gae_tau
    self.use_gae = config.use_gae
    self.lr_policy = config.optimizer_lr_policy
    self.lr_option = config.optimizer_lr_option
    self.mini_bs = config.mini_batch_size
    self.clip_eps = config.clip_eps
    self.lambda_entropy_policy = config.lambda_entropy_policy
    self.lambda_entropy_option = config.lambda_entropy_option

    self.policy = policy

    self.critic = OptionCritic(config,
                               dim_s=critic_dim,
                               dim_c=self.policy.dim_c)

    self.reset_optimizer()

  def reset_optimizer(self, lr_mult=1.0):
    self.optim_hi = torch.optim.Adam(self.critic.get_param() +
                                     self.policy.get_param(low_policy=False),
                                     lr=self.lr_option * lr_mult,
                                     weight_decay=1.e-3,
                                     eps=1e-5)
    self.optim_lo = torch.optim.Adam(self.critic.get_param() +
                                     self.policy.get_param(low_policy=True),
                                     lr=self.lr_policy * lr_mult,
                                     weight_decay=1.e-3,
                                     eps=1e-5)

  def calc_adv_each_episode(self, low_obs, actions, high_obs, prev_lat, latent,
                            reward):
    with torch.no_grad():
      vc = self.critic.get_value(high_obs)  # N x dim_c
      if self.train_option:
        pc = self.policy.log_trans(high_obs, prev_lat).exp()  # N x dim_c
        v_val_hi = (vc * pc).sum(dim=-1, keepdim=True).detach()
        pc = pc.detach()
        log_p_hi = self.policy.log_prob_option(high_obs, prev_lat,
                                               latent).detach()
      else:
        v_val_hi = torch.zeros_like(reward)
        pc = torch.zeros_like(reward)
        log_p_hi = torch.zeros_like(reward)

      if self.train_policy:
        v_val_lo = vc.gather(dim=-1, index=latent.long()).detach()
        log_p_lo = self.policy.log_prob_action(low_obs, latent,
                                               actions).detach()
      else:
        v_val_lo = torch.zeros_like(reward)
        log_p_lo = torch.zeros_like(reward)

      adv_hi = torch.zeros_like(reward)
      adv_lo = torch.zeros_like(reward)
      returns = torch.zeros_like(reward)
      next_value_hi = 0.
      next_value_lo = 0.
      adv_h = 0.
      adv_l = 0.
      ret = 0.

      for i in reversed(range(reward.size(0))):
        ret = reward[i] + self.gamma * ret
        returns[i] = ret

        if not self.use_gae:
          adv_hi[i] = ret - v_val_hi[i]
          adv_lo[i] = ret - v_val_lo[i]
        else:
          delta_hi = reward[i] + self.gamma * next_value_hi - v_val_hi[i]
          delta_lo = reward[i] + self.gamma * next_value_lo - v_val_lo[i]
          adv_h = delta_hi + self.gamma * self.gae_tau * adv_h
          adv_l = delta_lo + self.gamma * self.gae_tau * adv_l
          adv_hi[i], adv_lo[i] = adv_h, adv_l
          next_value_hi, next_value_lo = v_val_hi[i], v_val_lo[i]

    return returns, adv_hi, adv_lo, v_val_hi, v_val_lo, pc, log_p_hi, log_p_lo

  def step(self, low_obs, actions, high_obs, prev_lat, latent, returns, adv_hi,
           adv_lo, v_val_hi, v_val_lo, fixed_pc, fixed_logp_hi, fixed_logp_lo):
    n_batch = len(returns)

    if self.train_option:
      adv_hi = (adv_hi - adv_hi.mean()) / (adv_hi.std() +
                                           1e-8) if n_batch > 1 else 0.
      logp, entropy = self.policy.option_log_prob_entropy(
          high_obs, prev_lat, latent)
      vpred = (self.critic.get_value(high_obs) * fixed_pc).sum(dim=-1,
                                                               keepdim=True)

      vpred_clip = v_val_hi + (vpred - v_val_hi).clamp(-self.clip_eps,
                                                       self.clip_eps)
      vf_loss = torch.max((vpred - returns).square(),
                          (vpred_clip - returns).square()).mean()

      ratio = (logp - fixed_logp_hi).clamp_max(15.).exp()
      pg_loss = -torch.min(
          adv_hi * ratio,
          adv_hi *
          ratio.clamp(1.0 - self.clip_eps, 1.0 + self.clip_eps)).mean()
      loss = (pg_loss + vf_loss * 0.5 -
              self.lambda_entropy_option * entropy.mean())
      self.optim_hi.zero_grad()
      loss.backward()
      self.optim_hi.step()

    if self.train_policy:
      adv_lo = (adv_lo - adv_lo.mean()) / (adv_lo.std() +
                                           1e-8) if n_batch > 1 else 0.
      logp, entropy = self.policy.policy_log_prob_entropy(
          low_obs, latent, actions)
      vpred = self.critic.get_value(high_obs, latent)

      vpred_clip = v_val_lo + (vpred - v_val_lo).clamp(-self.clip_eps,
                                                       self.clip_eps)
      vf_loss = torch.max((vpred - returns).square(),
                          (vpred_clip - returns).square()).mean()

      ratio = (logp - fixed_logp_lo).clamp_max(15.).exp()
      pg_loss = -torch.min(
          adv_lo * ratio,
          adv_lo *
          ratio.clamp(1.0 - self.clip_eps, 1.0 + self.clip_eps)).mean()
      loss = (pg_loss + vf_loss * 0.5 -
              self.lambda_entropy_policy * entropy.mean())
      self.optim_lo.zero_grad()
      loss.backward()
      self.optim_lo.step()

import torch
import torch.nn.functional as F
from .option_policy import OptionPolicy, Policy
from .option_discriminator import (OptionDiscriminator, Discriminator)
from omegaconf import DictConfig


class GAIL(torch.nn.Module):

  def __init__(self, config: DictConfig, dim_s, dim_a, discrete_a):
    super(GAIL, self).__init__()
    self.dim_a = dim_a
    self.dim_s = dim_s
    self.device = torch.device(config.device)
    self.mini_bs = config.mini_batch_size
    lr = config.optimizer_lr_discriminator

    self.discriminator = Discriminator(config, dim_s=dim_s, dim_a=dim_a)
    self.policy = Policy(config,
                         dim_s=self.dim_s,
                         dim_a=self.dim_a,
                         discrete_a=discrete_a)

    self.optim = torch.optim.Adam(self.discriminator.parameters(),
                                  lr=lr,
                                  weight_decay=1.e-3)

    self.to(self.device)

  def gail_reward(self, obs, action):
    d = self.discriminator.get_unnormed_d(obs, action)
    reward = -F.logsigmoid(d)
    return reward

  def step(self, obs, action, is_policy):
    src = self.discriminator.get_unnormed_d(obs, action)
    loss = F.binary_cross_entropy_with_logits(src, is_policy)
    self.optim.zero_grad()
    loss.backward()
    self.optim.step()


class OptionGAIL(torch.nn.Module):

  def __init__(self,
               config: DictConfig,
               dim_disc_obs,
               dim_low_obs,
               dim_high_obs,
               dim_a,
               dim_c,
               discrete_a=False):
    super(OptionGAIL, self).__init__()
    self.with_c = config.use_c_in_discriminator
    self.mini_bs = config.mini_batch_size
    self.use_d_info_gail = config.use_d_info_gail
    self.device = torch.device(config.device)

    self.discriminator = OptionDiscriminator(config,
                                             dim_s=dim_disc_obs,
                                             dim_a=dim_a,
                                             dim_c=dim_c)
    self.policy = OptionPolicy(config,
                               dim_low_obs=dim_low_obs,
                               dim_high_obs=dim_high_obs,
                               dim_a=dim_a,
                               dim_c=dim_c,
                               discrete_a=discrete_a)

    self.optim = torch.optim.Adam(self.discriminator.parameters(),
                                  weight_decay=1.e-3)

    self.to(self.device)

  def gail_reward(self, disc_obs, prev_lat, action, latent):
    d = self.discriminator.get_unnormed_d(disc_obs, prev_lat, action, latent)
    reward = -F.logsigmoid(d)
    return reward

  def step(self, disc_obs, prev_lat, action, latent, is_policy):
    src = self.discriminator.get_unnormed_d(disc_obs, prev_lat, action, latent)
    loss = F.binary_cross_entropy_with_logits(src, is_policy)
    self.optim.zero_grad()
    loss.backward()
    self.optim.step()

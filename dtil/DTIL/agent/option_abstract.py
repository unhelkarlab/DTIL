import abc
import torch
import numpy as np
from omegaconf import DictConfig
from ...helper.utils import conv_input, conv_tuple_input


class AbstractPolicyLeaner(abc.ABC):

  def __init__(self, config: DictConfig):
    self.gamma = config.gamma
    self.device = torch.device(config.device)
    self.actor = None
    self.clip_grad_val = config.clip_grad_val
    self.num_critic_update = config.num_critic_update
    self.num_actor_update = config.num_actor_update

  def conv_input(self, batch_input, is_onehot_needed, dimension):
    return conv_input(batch_input, is_onehot_needed, dimension, self.device)

  def conv_tuple_input(self, tup_batch, tup_is_onehot_needed, tup_dimension):
    return conv_tuple_input(tup_batch, tup_is_onehot_needed, tup_dimension,
                            self.device)

  @abc.abstractmethod
  def reset_optimizers(self, config: DictConfig):
    pass

  @abc.abstractmethod
  def train(self, training=True):
    pass

  @property
  @abc.abstractmethod
  def alpha(self):
    pass

  @property
  @abc.abstractmethod
  def critic_net(self):
    pass

  @property
  @abc.abstractmethod
  def critic_target_net(self):
    pass

  @abc.abstractmethod
  def choose_action(self, tup_obs, option, sample=False):
    pass

  @abc.abstractmethod
  def critic(self, tup_obs, option, action, both=False):
    pass

  @abc.abstractmethod
  def getV(self, tup_obs, option):
    pass

  @abc.abstractmethod
  def get_targetV(self, tup_obs, option):
    pass

  @abc.abstractmethod
  def update(self, tup_obs, option, action, tup_next_obs, next_option, reward,
             done, logger, step):
    pass

  @abc.abstractmethod
  def update_critic(self, tup_obs, option, action, tup_next_obs, next_option,
                    reward, done, logger, step):
    pass

  @abc.abstractmethod
  def save(self, path, suffix=""):
    pass

  @abc.abstractmethod
  def load(self, path):
    pass

  @abc.abstractmethod
  def log_probs(self, tup_obs, action):
    pass

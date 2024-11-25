import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import distributions as pyd
from torch.distributions import Categorical
from ...helper.transformed_distributions import SquashedNormal


def weight_init(m):
  """Custom weight init for Conv2D and Linear layers."""
  if isinstance(m, nn.Linear):
    nn.init.orthogonal_(m.weight.data)
    if hasattr(m.bias, 'data'):
      m.bias.data.fill_(0.0)


def mlp(input_dim, output_dim, list_hidden_dims, output_mod=None):
  if len(list_hidden_dims) == 0:
    mods = [nn.Linear(input_dim, output_dim)]
  else:
    mods = [nn.Linear(input_dim, list_hidden_dims[0]), nn.ReLU(inplace=True)]
    for i in range(len(list_hidden_dims) - 1):
      mods += [
          nn.Linear(list_hidden_dims[i], list_hidden_dims[i + 1]),
          nn.ReLU(inplace=True)
      ]
    mods.append(nn.Linear(list_hidden_dims[-1], output_dim))
  if output_mod is not None:
    mods.append(output_mod)
  trunk = nn.Sequential(*mods)
  return trunk


class AbstractActor(nn.Module):

  def __init__(self):
    super().__init__()

  def forward(self, obs):
    raise NotImplementedError

  def rsample(self, obs):
    raise NotImplementedError

  def sample(self, obs):
    raise NotImplementedError

  def exploit(self, obs):
    raise NotImplementedError

  def is_discrete(self):
    raise NotImplementedError


class DiagGaussianActor(AbstractActor):
  """torch.distributions implementation of an diagonal Gaussian policy."""

  def __init__(self,
               obs_dim,
               action_dim,
               list_hidden_dims,
               log_std_bounds,
               bounded=True,
               use_nn_logstd=False,
               clamp_action_logstd=False):
    super().__init__()
    self.use_nn_logstd = use_nn_logstd
    self.clamp_action_logstd = clamp_action_logstd

    output_dim = action_dim
    if self.use_nn_logstd:
      output_dim = 2 * action_dim
    else:
      self.action_logstd = nn.Parameter(
          torch.empty(1, action_dim, dtype=torch.float32).fill_(0.))

    self.trunk = mlp(obs_dim, output_dim, list_hidden_dims)

    self.apply(weight_init)

    self.log_std_bounds = log_std_bounds
    self.bounded = bounded

  def forward(self, obs):
    if self.use_nn_logstd:
      mu, log_std = self.trunk(obs).chunk(2, dim=-1)
    else:
      mu = self.trunk(obs)
      log_std = self.action_logstd.expand_as(mu)

    # clamp logstd
    if self.clamp_action_logstd:
      log_std = log_std.clamp(self.log_std_bounds[0], self.log_std_bounds[1])
    else:
      # constrain log_std inside [log_std_min, log_std_max]
      log_std = torch.tanh(log_std)
      log_std_min, log_std_max = self.log_std_bounds
      log_std = log_std_min + 0.5 * (log_std_max - log_std_min) * (log_std + 1)
    std = log_std.exp()

    if self.bounded:
      dist = SquashedNormal(mu, std)
    else:
      mu = mu.clamp(-10, 10)
      dist = pyd.Normal(mu, std)

    return dist

  def rsample(self, obs):
    dist = self.forward(obs)
    action = dist.rsample()
    log_prob = dist.log_prob(action).sum(-1, keepdim=True)

    return action, log_prob

  def sample(self, obs):
    return self.rsample(obs)

  def exploit(self, obs):
    return self.forward(obs).mean

  def is_discrete(self):
    return False


class DiscreteActor(AbstractActor):
  'cf) https://github.com/openai/spinningup/issues/148 '

  def __init__(self, obs_dim, action_dim, list_hidden_dims):
    super().__init__()

    output_dim = action_dim
    self.trunk = mlp(obs_dim, output_dim, list_hidden_dims)

    self.apply(weight_init)

  def forward(self, obs):
    logits = self.trunk(obs)
    dist = Categorical(logits=logits)
    return dist

  def action_probs(self, obs):
    dist = self.forward(obs)
    action_probs = dist.probs
    # avoid numerical instability
    z = (action_probs == 0.0).float() * 1e-10
    log_action_probs = torch.log(action_probs + z)

    return action_probs, log_action_probs

  def exploit(self, obs):
    dist = self.forward(obs)
    return dist.logits.argmax(dim=-1)

  def sample(self, obs):
    dist = self.forward(obs)

    samples = dist.sample()
    action_log_probs = dist.log_prob(samples)

    return samples, action_log_probs

  # TODO: temporary solution
  def sample_w_avail_actions(self, obs, avail_actions=None):
    logits = self.trunk(obs)
    if avail_actions is not None:
      logits[avail_actions.reshape(logits.shape) == 0] = -1e10

    probs = F.softmax(logits, dim=-1)
    dist = Categorical(probs=probs)

    samples = dist.sample()

    return samples

  def rsample(self, obs):
    'should not be used'
    raise NotImplementedError

  def is_discrete(self):
    return True

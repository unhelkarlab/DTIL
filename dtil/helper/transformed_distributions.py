import torch
import math
import torch.nn.functional as F
from torch import distributions as pyd
from torch.distributions import RelaxedOneHotCategorical


class TanhTransform(pyd.transforms.Transform):
  domain = pyd.constraints.real
  codomain = pyd.constraints.interval(-1.0, 1.0)
  bijective = True
  sign = +1

  def __init__(self, cache_size=1):
    super().__init__(cache_size=cache_size)

  @staticmethod
  def atanh(x):
    return 0.5 * (x.log1p() - (-x).log1p())

  def __eq__(self, other):
    return isinstance(other, TanhTransform)

  def _call(self, x):
    return x.tanh()

  def _inverse(self, y):
    # We do not clamp to the boundary here as it may degrade the performance of certain algorithms.
    # one should use `cache_size=1` instead
    return self.atanh(y)

  def log_abs_det_jacobian(self, x, y):
    # We use a formula that is more numerically stable, see details in the following link
    # https://github.com/tensorflow/probability/commit/ef6bb176e0ebd1cf6e25c6b5cecdd2428c22963f#diff-e120f70e92e6741bca649f04fcd907b7
    return 2. * (math.log(2.) - x - F.softplus(-2. * x))


class SquashedNormal(pyd.transformed_distribution.TransformedDistribution):

  def __init__(self, loc, scale):
    self.loc = loc
    self.scale = scale

    self.base_dist = pyd.Normal(loc, scale)
    transforms = [TanhTransform()]
    super().__init__(self.base_dist, transforms)

  @property
  def mean(self):
    mu = self.loc
    for tr in self.transforms:
      mu = tr(mu)
    return mu


class GumbelSoftmax(RelaxedOneHotCategorical):
  '''
    A differentiable Categorical distribution using reparametrization trick with Gumbel-Softmax
    Explanation http://amid.fish/assets/gumbel.html
    NOTE: use this in place PyTorch's RelaxedOneHotCategorical distribution since its log_prob is not working right (returns positive values)
    Papers:
    [1] The Concrete Distribution: A Continuous Relaxation of Discrete Random Variables (Maddison et al, 2017)
    [2] Categorical Reparametrization with Gumbel-Softmax (Jang et al, 2017)
    '''

  def sample(self, sample_shape=torch.Size()):
    '''Gumbel-softmax sampling. Note rsample is inherited from RelaxedOneHotCategorical'''
    u = torch.empty(self.logits.size(),
                    device=self.logits.device,
                    dtype=self.logits.dtype).uniform_(0, 1)
    noisy_logits = self.logits - torch.log(-torch.log(u))
    return torch.argmax(noisy_logits, dim=-1)

  def rsample(self, sample_shape=torch.Size()):
    '''
      ref: https://github.com/kengz/SLM-Lab/blob/master/slm_lab/lib/distribution.py
      Gumbel-softmax resampling using the Straight-Through trick.
      Credit to Ian Temple for bringing this to our attention. To see standalone code of how this works, refer to https://gist.github.com/yzh119/fd2146d2aeb329d067568a493b20172f
      '''
    rout = super().rsample(sample_shape)  # differentiable
    out = F.one_hot(torch.argmax(rout, dim=-1), self.logits.shape[-1]).float()
    return (out - rout).detach() + rout

  def log_prob(self, value):
    '''value is one-hot or relaxed'''
    if value.shape != self.logits.shape:
      value = F.one_hot(value.long(), self.logits.shape[-1]).float()
      assert value.shape == self.logits.shape
    return -torch.sum(-value * F.log_softmax(self.logits, -1), -1)

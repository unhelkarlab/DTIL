from .mahil import MAHIL
from omegaconf import DictConfig
from pettingzoo.utils.env import ParallelEnv
from gymnasium.spaces import Discrete, Box
from gym.spaces import Discrete as GymDiscrete
from gym.spaces import Box as GymBox


def make_mahil_agent(config: DictConfig, env: ParallelEnv, agent_idx):

  agent_name = env.agents[agent_idx]
  latent_dim = config.dim_c[agent_idx]
  obs_space = env.observation_space(agent_name)
  if isinstance(obs_space, Discrete) or isinstance(obs_space, GymDiscrete):
    obs_dim = obs_space.n
    discrete_obs = True
  else:
    obs_dim = obs_space.shape[0]
    discrete_obs = False

  list_aux_dim = []
  list_discrete_aux = []
  for name in env.agents:
    act_space = env.action_space(name)
    if not (isinstance(act_space, Discrete)
            or isinstance(act_space, GymDiscrete) or isinstance(act_space, Box)
            or isinstance(act_space, GymBox)):
      raise RuntimeError(
          "Invalid action space: Only Discrete and Box action spaces supported")

    if isinstance(act_space, Discrete) or isinstance(act_space, GymDiscrete):
      tmp_action_dim = act_space.n
      tmp_discrete_act = True
    else:
      tmp_action_dim = act_space.shape[0]
      tmp_discrete_act = False

    if name == agent_name:
      action_dim = tmp_action_dim
      discrete_act = tmp_discrete_act

    if config.use_auxiliary_obs:
      list_aux_dim.append(tmp_action_dim)
      list_discrete_aux.append(tmp_discrete_act)

  agent = MAHIL(config, obs_dim, action_dim, latent_dim, tuple(list_aux_dim),
                discrete_obs, discrete_act, tuple(list_discrete_aux))
  return agent

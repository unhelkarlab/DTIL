from pettingzoo_domain.po_movers_v2 import PO_Movers_V2
from pettingzoo_domain.po_flood_v2 import PO_Flood_V2
from pettingzoo_domain.labor_division import (TwoTargetDyadLaborDivision,
                                              ThreeTargetDyadLaborDivision)
from pettingzoo_domain.labor_division_v2 import (TwoTargetDyadLaborDivisionV2,
                                                 ThreeTargetDyadLaborDivisionV2)
from pettingzoo_domain.multi_subtasks import (MultiSubTasksDyadTwoTargets,
                                              MultiSubTasksDyadThreeTargets)
from pettingzoo.mpe import (simple_crypto_v3, simple_push_v3,
                            simple_adversary_v3, simple_speaker_listener_v4,
                            simple_spread_v3, simple_tag_v3)
from pettingzoo_domain.conv_gym_domain import ConvGymDomain
from pettingzoo_domain.overcooked import Overcooked
import gym
from stable_baselines3.common.monitor import Monitor
import gym_custom


def env_generator(config):
  '''
    return:
      fn_env_factory: a factory function that creates a pettingzoo env
      env_kwargs: a dictionary of kwargs for the env
  '''
  env_name = config.env_name
  if env_name == "PO_Movers-v2":
    return PO_Movers_V2, {}
  elif env_name == "PO_Flood-v2":
    return PO_Flood_V2, {}
  elif env_name in ["Protoss5v5", "Terran5v5"]:
    import pettingzoo_domain.smac_v2_env as smac_v2_env
    if env_name == "Protoss5v5":
      return smac_v2_env.Protoss5v5, {}
    elif env_name == "Terran5v5":
      return smac_v2_env.Terran5v5, {}
  elif env_name == "MultiGoals2D_2-v0":
    return ConvGymDomain, {"env_name": env_name}
  elif env_name == "LaborDivision2":
    return TwoTargetDyadLaborDivision, {}
  elif env_name == "LaborDivision3":
    return ThreeTargetDyadLaborDivision, {}
  elif env_name == "LaborDivision2-v2":
    return TwoTargetDyadLaborDivisionV2, {}
  elif env_name == "LaborDivision3-v2":
    return ThreeTargetDyadLaborDivisionV2, {}
  elif env_name == "MultiSubTasks2":
    return MultiSubTasksDyadTwoTargets, {}
  elif env_name == "MultiSubTasks3":
    return MultiSubTasksDyadThreeTargets, {}
  elif env_name == "overcooked":
    return Overcooked, {}
  elif env_name[:4] == "sc2_":
    from pettingzoo_domain.smac_v1_env import SMAC_V1
    return SMAC_V1, {"map_name": env_name[4:]}
  # Multi Particle Environments (MPE)
  elif env_name == "simple_crypto":
    kwargs = {"continuous_actions": False}
    return simple_crypto_v3.parallel_env, kwargs
  elif env_name == "simple_push":
    kwargs = {"continuous_actions": False}
    return simple_push_v3.parallel_env, kwargs
  elif env_name == "simple_adversary":
    kwargs = {"continuous_actions": False}
    return simple_adversary_v3.parallel_env, kwargs
  elif env_name == "simple_speaker_listener":
    kwargs = {"continuous_actions": False}
    return simple_speaker_listener_v4.parallel_env, kwargs
  elif env_name == "simple_spread":
    kwargs = {"continuous_actions": False}
    return simple_spread_v3.parallel_env, kwargs
  elif env_name == "simple_tag":
    kwargs = {"continuous_actions": False}
    return simple_tag_v3.parallel_env, kwargs

from dtil.pettingzoo_envs.po_movers_v2 import PO_Movers_V2
from dtil.pettingzoo_envs.po_flood_v2 import PO_Flood_V2
from dtil.pettingzoo_envs.multi_jobs import (TwoTargetDyadMultiJobs,
                                             ThreeTargetDyadMultiJobs)


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
    import dtil.pettingzoo_envs.smac_v2_env as smac_v2_env
    if env_name == "Protoss5v5":
      return smac_v2_env.Protoss5v5, {}
    elif env_name == "Terran5v5":
      return smac_v2_env.Terran5v5, {}
  elif env_name == "MultiJobs2":
    return TwoTargetDyadMultiJobs, {}
  elif env_name == "MultiJobs3":
    return ThreeTargetDyadMultiJobs, {}

import os
import hydra
import datetime
import time
from omegaconf import OmegaConf, DictConfig
import env_generator as envgen


def get_dirs(base_dir, alg_name, env_name, msg="default"):

  base_log_dir = os.path.join(base_dir, "result/")

  ts_str = datetime.datetime.fromtimestamp(
      time.time()).strftime("%Y-%m-%d_%H-%M-%S")
  log_dir_root = os.path.join(base_log_dir, env_name, alg_name, msg, ts_str)
  save_dir = os.path.join(log_dir_root, "model")
  log_dir = os.path.join(log_dir_root, "log")
  os.makedirs(save_dir)
  os.makedirs(log_dir)

  return log_dir, save_dir


def run_alg(config):
  alg_name = config.alg_name
  msg = f"{config.tag}"
  log_interval = config.log_interval
  eval_interval = config.eval_interval

  log_dir, output_dir = get_dirs(config.base_dir, alg_name, config.env_name,
                                 msg)
  pretrain_name = os.path.join(config.base_dir, config.pretrain_path)

  # fix dim_c to 1 for independent iql
  if alg_name == "iiql":
    config.dim_c = [1] * len(config.dim_c)

  # save config
  config_path = os.path.join(log_dir, "config.yaml")
  with open(config_path, "w") as outfile:
    OmegaConf.save(config=config, f=outfile)

  if (config.data_path.endswith("torch") or config.data_path.endswith("pt")
      or config.data_path.endswith("pkl") or config.data_path.endswith("npy")):
    demo_path = os.path.join(config.base_dir, config.data_path)
  else:
    print(f"Data path not exists: {config.data_path}")

  fn_env_factory, env_kwargs = envgen.env_generator(config)

  if alg_name == "mahil" or alg_name == "iiql":
    from aic_ml.MAHIL.train_mahil import train
    train(config, demo_path, log_dir, output_dir, fn_env_factory, log_interval,
          eval_interval, env_kwargs)
  elif alg_name == "maogail":
    from aic_ml.baselines.ma_ogail.train_ma_ogail import learn
    learn(config, True, demo_path, log_dir, output_dir, fn_env_factory,
          pretrain_name, eval_interval, env_kwargs)
  elif alg_name == "magail":
    from aic_ml.baselines.ma_ogail.train_ma_ogail import learn
    learn(config, False, demo_path, log_dir, output_dir, fn_env_factory,
          pretrain_name, eval_interval, env_kwargs)
  # add (multi-agent) bc
  elif alg_name == "bc":
    from aic_ml.baselines.bc import train_bc
    train_bc(config, demo_path, log_dir, output_dir, fn_env_factory,
             log_interval, eval_interval, env_kwargs)


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
  import torch.multiprocessing as multiprocessing
  multiprocessing.set_start_method('spawn')

  cur_dir = os.path.dirname(__file__)
  cfg.base_dir = cur_dir
  print(OmegaConf.to_yaml(cfg))

  run_alg(cfg)


if __name__ == "__main__":
  start_time = time.time()
  main()
  print("Excution time: ", time.time() - start_time)

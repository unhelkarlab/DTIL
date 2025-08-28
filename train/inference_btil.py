import numpy as np
from omegaconf import OmegaConf
import torch
import os
import pandas as pd
from dtil.helper.utils import load_trajectories, evaluate
from dtil.pettingzoo_envs.po_movers_v2 import PO_Movers_V2
from dtil.pettingzoo_envs.po_flood_v2 import PO_Flood_V2
import run_btil
import inference


def load_btil_agent(model_dir, env_name, agent_idx, model_num, supervision):
  prefix = env_name + "_btil_svi_"
  postfix = f"_a{agent_idx + 1}_s{model_num}"

  policy_name = prefix + "policy" + postfix
  tx_name = prefix + "tx" + postfix
  bx_name = prefix + "bx" + postfix

  pi = np.load(os.path.join(model_dir, policy_name + ".npy"))
  tx = np.load(os.path.join(model_dir, tx_name + ".npy"))
  bx = np.load(os.path.join(model_dir, bx_name + ".npy"))

  # pi: X x O x A
  # tx: X x O x X
  # bx: O x X
  return pi, tx, bx


def infer_mental_states(states, actions, pi, tx, bx):
  len_traj = len(states)
  n_x = len(pi)

  log_pis = np.zeros((len_traj, 1, n_x))
  log_txs = np.zeros((len_traj, n_x, n_x))
  for i in range(len_traj):
    stt = states[i]
    act = actions[i][0]
    log_pis[i] = np.log(pi[:, stt, act])
    log_txs[i] = np.log(tx[:, stt, :])

  log_prob = log_txs + log_pis
  log_prob0 = np.log(bx[states[0]]) + log_pis[0, 0]

  # forward
  max_path = np.zeros((len_traj, n_x))
  accumulate_logp = log_prob0
  max_path[0] = n_x
  for idx in range(1, len_traj):
    logp_xx = accumulate_logp[..., np.newaxis] + log_prob[idx]
    accumulate_logp = logp_xx.max(axis=-2)
    max_path[idx, :] = logp_xx.argmax(axis=-2)

  # backward
  x_array = np.zeros((len_traj + 1, ))
  x_array[-1] = accumulate_logp.argmax(axis=-1)
  for idx in range(len_traj, 0, -1):
    x_array[idx - 1] = max_path[idx - 1, int(x_array[idx])]

  return list(x_array[1:])


def infer_latent_result_btil(model_dir, env_name, agent_idx, supervision,
                             model_num, trajectories):

  pi, tx, bx = load_btil_agent(model_dir, env_name, agent_idx, model_num,
                               supervision)

  agent_trajs = trajectories[agent_idx]

  list_inferred_x = []
  list_true_x = []
  for i_e in range(len(agent_trajs)):
    states, actions, latents = list(zip(*agent_trajs[i_e]))
    true_x = [lat[0] for lat in latents]
    list_true_x.append(true_x)

    inferred_x = infer_mental_states(states, actions, pi, tx, bx)
    list_inferred_x.append(inferred_x)

  ham_dists, lengths = inference.get_stats_about_x(list_inferred_x, list_true_x)
  accuracy = 1 - np.sum(ham_dists) / np.sum(lengths)

  return accuracy


class BTIL_Agent:

  def __init__(self, pi, tx, bx,
               converter: run_btil.Converter_for_BTIL) -> None:
    self.pi = pi
    self.tx = tx
    self.bx = bx
    self.converter = converter

    self.num_latent = len(self.pi)
    self.PREV_LATENT = len(self.pi)
    self.PREV_AUX = None

  def choose_action(self, obs, prev_latent, prev_aux, sample, avail_actions):
    sidx = self.converter.convert_obs_2_sidx(obs)

    if prev_latent == self.PREV_LATENT:
      latent = np.random.choice(range(self.num_latent), p=self.bx[sidx])
    else:
      latent = np.random.choice(range(self.num_latent),
                                p=self.tx[prev_latent, sidx])

    num_actions = self.pi.shape[-1]
    action = np.random.choice(range(num_actions), p=self.pi[latent, sidx])

    return latent, action


if __name__ == "__main__":
  INFERENCE = False
  EVAL_REWARD = True

  cur_dir = os.path.dirname(__file__)
  model_dir = os.path.join(cur_dir, "result/btil_models/")
  if EVAL_REWARD:
    env_name = "PO_Movers-v2"
    if env_name == "PO_Movers-v2":
      env = PO_Movers_V2()
      converter = run_btil.Converter_PO_Movers()
    elif env_name == "PO_Flood-v2":
      env = PO_Flood_V2()
      converter = run_btil.Converter_PO_Flood()
    else:
      raise NotImplementedError()

    seed = 1
    supervision = 0.2
    a1_pi, a1_tx, a1_bx = load_btil_agent(model_dir, env_name, 0, seed,
                                          supervision)
    a2_pi, a2_tx, a2_bx = load_btil_agent(model_dir, env_name, 1, seed,
                                          supervision)
    dict_agents = {
        0: BTIL_Agent(a1_pi, a1_tx, a1_bx, converter),
        1: BTIL_Agent(a2_pi, a2_tx, a2_bx, converter)
    }

    env.reset()
    total_returns, total_timesteps, wins = evaluate(dict_agents, env, False, 10)
    print(np.mean(total_returns[0]), np.std(total_returns[0]))

  if INFERENCE:
    env_names = ["PO_Movers-v2", "PO_Flood-v2"]
    alg_supervisions = [("btil", 0.2)]
    agent_idxs = [0, 1]
    model_numbers = [1, 2, 3]

    columns = ["env", "alg", "sv", "agent_idx", "model_num", "accuracy"]

    list_results = []
    for env_name in env_names:
      test_data_dir = os.path.join(cur_dir, "test_data/")
      trajectories = inference.load_test_data(test_data_dir, env_name)
      if env_name == "PO_Movers-v2":
        converter = run_btil.Converter_PO_Movers()
      elif env_name == "PO_Flood-v2":
        converter = run_btil.Converter_PO_Flood()
      else:
        raise NotImplementedError()

      btil_trajs = converter.convert_trajectories(trajectories, None)

      for alg_name, supervision in alg_supervisions:
        for agent_idx in agent_idxs:
          for model_num in model_numbers:
            accuracy = infer_latent_result_btil(model_dir, env_name, agent_idx,
                                                supervision, model_num,
                                                btil_trajs)
            list_results.append((env_name, alg_name, supervision, agent_idx,
                                 model_num, accuracy))

    df = pd.DataFrame(list_results, columns=columns)

    cur_dir = os.path.dirname(__file__)
    save_name = os.path.join(cur_dir, "infer_latent_result_btil2.csv")
    df.to_csv(save_name, index=False)

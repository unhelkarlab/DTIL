import numpy as np
from omegaconf import OmegaConf
import torch
import os
import pandas as pd
from dtil.helper.utils import load_trajectories


def hamming_distance(seq1, seq2):
  assert len(seq1) == len(seq2)

  count = 0
  for idx, elem in enumerate(seq1):
    if elem != seq2[idx]:
      count += 1

  return count


def get_stats_about_x(list_inferred_x, list_true_x):
  dis_array = []
  length_array = []
  for i_e, inferred_x in enumerate(list_inferred_x):
    res = hamming_distance(inferred_x, list_true_x[i_e])
    dis_array.append(res)
    length_array.append(len(inferred_x))

  dis_array = np.array(dis_array)
  length_array = np.array(length_array)
  return dis_array, length_array


def infer_latent_result(dict_agents, list_agent_names, agent_idx, trajectories):

  agent = dict_agents[list_agent_names[agent_idx]]

  agent_trajs = trajectories[agent_idx]

  list_inferred_x = []
  for i_e in range(len(agent_trajs["states"])):
    states = agent_trajs["states"][i_e]
    actions = agent_trajs["actions"][i_e]
    inferred_x, _ = agent.infer_mental_states(states, actions, [])
    list_inferred_x.append(inferred_x)

  ham_dists, lengths = get_stats_about_x(list_inferred_x,
                                         agent_trajs["latents"])
  accuracy = 1 - np.sum(ham_dists) / np.sum(lengths)

  return accuracy


def load_test_data(data_dir, env_name):
  if env_name == "MultiJobs2":
    data_path = os.path.join(data_dir, "MultiJobs2_100.pkl")
  elif env_name == "MultiJobs3":
    data_path = os.path.join(data_dir, "MultiJobs3_100.pkl")
  elif env_name == "PO_Flood-v2":
    data_path = os.path.join(data_dir, "PO_Flood-v2_100.pkl")
  elif env_name == "PO_Movers-v2":
    data_path = os.path.join(data_dir, "PO_Movers-v2_100.pkl")
  else:
    raise ValueError(f"Unknown env_name: {env_name}")

  trajectories = load_trajectories(data_path, 100)

  return trajectories

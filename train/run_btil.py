import abc
import os
import numpy as np
import itertools
import pickle
import click
from aic_core.utils.mdp_utils import StateSpace
from aic_domain.box_push_v2 import AGENT_ACTIONSPACE as MOVERS_ACTIONSPACE
from aic_domain.rescue import AGENT_ACTIONSPACE as FLOOD_ACTIONSPACE
from pettingzoo_domain.po_movers_v2 import PO_Movers_V2
from pettingzoo_domain.po_flood_v2 import PO_Flood_V2
from aic_ml.MAHIL.helper.utils import load_trajectories
from aic_ml.BTIL.btil_svi import BTIL_SVI


def one_hot(n, dim):
  np_oh = np.zeros(dim)
  np_oh[n] = 1
  return np_oh


class Converter_for_BTIL(abc.ABC):

  def __init__(self) -> None:

    self.init_statespace()
    self.init_statespace_helper_vars()

  @abc.abstractmethod
  def init_statespace(self):
    self.dict_factored_statespace = {}
    self.dummy_states = None
    pass

  @abc.abstractmethod
  def convert_obs_2_sidx(self, obs):
    pass

  @abc.abstractmethod
  def convert_sidx_2_obs(self, sidx):
    pass

  def init_statespace_helper_vars(self):

    # Retrieve number of states and state factors.
    self.num_state_factors = len(self.dict_factored_statespace)
    self.list_num_states = []
    for idx in range(self.num_state_factors):
      self.list_num_states.append(
          self.dict_factored_statespace.get(idx).num_states)

    self.num_actual_states = np.prod(self.list_num_states)
    self.num_dummy_states = (0 if self.dummy_states is None else
                             self.dummy_states.num_states)
    self.num_states = self.num_actual_states + self.num_dummy_states

    # Create mapping from state to state index.
    # Mapping takes state value as inputs and outputs a scalar state index.
    np_list_idx = np.arange(self.num_actual_states, dtype=np.int32)
    self.np_state_to_idx = np_list_idx.reshape(self.list_num_states)

    # Create mapping from state index to state.
    # Mapping takes state index as input and outputs a factored state.
    np_idx_to_state = np.zeros((self.num_actual_states, self.num_state_factors),
                               dtype=np.int32)
    for state, idx in np.ndenumerate(self.np_state_to_idx):
      np_idx_to_state[idx] = state
    self.np_idx_to_state = np_idx_to_state

  def convert_trajectories(self, trajectories, n_labeled=None):

    btil_trajs = []
    for each_agent_trajs in trajectories:
      converted_trajs = []

      n_epi = len(each_agent_trajs["states"])
      for i_e in range(n_epi):
        len_epi = each_agent_trajs["lengths"][i_e]

        btil_states = []
        btil_actions = []
        btil_latents = []
        for i_t in range(len_epi):
          stt = each_agent_trajs["states"][i_e][i_t]
          sidx = self.convert_obs_2_sidx(stt)
          btil_states.append(sidx)

          act = each_agent_trajs["actions"][i_e][i_t]
          btil_actions.append((act, ))

          if n_labeled is None or i_e < n_labeled:
            lat = each_agent_trajs["latents"][i_e][i_t]
            btil_latents.append((lat, ))
          else:
            btil_latents.append((None, ))

        btil_epi = list(zip(btil_states, btil_actions, btil_latents))
        converted_trajs.append(btil_epi)

      btil_trajs.append(converted_trajs)

    return btil_trajs


class Converter_PO_Movers(Converter_for_BTIL):

  def init_statespace(self):
    self.dict_factored_statespace = {}

    possible_coords = [(x, y) for x in range(7) for y in range(7)]

    self.my_pos_space = StateSpace(possible_coords)

    action_idxs = list(range(MOVERS_ACTIONSPACE.num_actions))
    self.my_action_space = StateSpace(action_idxs)

    possible_otherplayer_state = [None]
    for x in [-1, 0, 1]:
      for y in [-1, 0, 1]:
        for act in action_idxs:
          possible_otherplayer_state.append((x, y, act))
    self.otherplayer_space = StateSpace(possible_otherplayer_state)

    box_states = []
    for item in itertools.product(range(4), repeat=3):
      box_states.append(item)

    self.box_space = StateSpace(box_states)

    self.dict_factored_statespace = {
        0: self.my_pos_space,
        1: self.my_action_space,
        2: self.otherplayer_space,
        3: self.box_space
    }

    self.dummy_states = None

  def convert_obs_2_sidx(self, obs):
    my_x = np.where(obs[0:7] == 1)[0][0]
    my_y = np.where(obs[7:14] == 1)[0][0]
    my_a = np.where(obs[14:20] == 1)[0][0]

    fr_o = bool(obs[20])

    box1 = np.where(obs[33:37] == 1)[0][0]
    box2 = np.where(obs[37:41] == 1)[0][0]
    box3 = np.where(obs[41:45] == 1)[0][0]

    mypos_idx = self.my_pos_space.state_to_idx[(my_x, my_y)]
    myact_idx = self.my_action_space.state_to_idx[my_a]
    if fr_o:
      fr_x = np.where(obs[21:24] == 1)[0][0] - 1
      fr_y = np.where(obs[24:27] == 1)[0][0] - 1
      fr_a = np.where(obs[27:33] == 1)[0][0]
      otherplayer_idx = self.otherplayer_space.state_to_idx[(fr_x, fr_y, fr_a)]
    else:
      otherplayer_idx = self.otherplayer_space.state_to_idx[None]

    box_idx = self.box_space.state_to_idx[(box1, box2, box3)]

    sidx = self.np_state_to_idx[(mypos_idx, myact_idx, otherplayer_idx,
                                 box_idx)]
    return sidx

  def convert_sidx_2_obs(self, sidx):
    vec_state = self.np_idx_to_state[sidx]
    mypos_idx = vec_state[0]
    mypos = self.my_pos_space.idx_to_state[mypos_idx]
    myact_idx = vec_state[1]
    myact = self.my_action_space.idx_to_state[myact_idx]
    otherplayer_idx = vec_state[2]
    otherplayer = self.otherplayer_space.idx_to_state[otherplayer_idx]
    box_idx = vec_state[3]
    box = self.box_space.idx_to_state[box_idx]

    my_x = one_hot(mypos[0], 7)
    my_y = one_hot(mypos[1], 7)
    my_a = one_hot(myact, 6)
    if otherplayer is None:
      fr_o = np.zeros(1)
      fr_x = np.zeros(3)
      fr_y = np.zeros(3)
      fr_a = np.zeros(6)
    else:
      fr_o = np.ones(1)
      fr_x = one_hot(otherplayer[0] + 1, 3)
      fr_y = one_hot(otherplayer[1] + 1, 3)
      fr_a = one_hot(otherplayer[2], 6)

    box1 = one_hot(box[0], 4)
    box2 = one_hot(box[1], 4)
    box3 = one_hot(box[2], 4)

    obs = np.concatenate(
        [my_x, my_y, my_a, fr_o, fr_x, fr_y, fr_a, box1, box2, box3])
    return obs


class Converter_PO_Flood(Converter_for_BTIL):

  def __init__(self) -> None:
    self.env = PO_Flood_V2(None)
    super().__init__()

  def init_statespace(self):
    self.dict_factored_statespace = {}

    num_loc = self.env.mmdp.pos1_space.num_states

    self.my_pos_space = StateSpace(list(range(num_loc)))

    action_idxs = list(range(FLOOD_ACTIONSPACE.num_actions))
    self.my_action_space = StateSpace(action_idxs)

    possible_op_state = [None]
    for loc_idx in range(len(self.env.possible_locations)):
      for act in action_idxs:
        possible_op_state.append((loc_idx, act))

    self.otherplayer_space = StateSpace(possible_op_state)

    possible_work_states = []
    for item in itertools.product(range(2), repeat=4):
      possible_work_states.append(item)

    self.work_space = StateSpace(possible_work_states)

    self.dict_factored_statespace = {
        0: self.my_pos_space,
        1: self.my_action_space,
        2: self.otherplayer_space,
        3: self.work_space
    }

    self.dummy_states = None

  def convert_obs_2_sidx(self, obs):
    my_loc = np.where(obs[0:32] == 1)[0][0]
    my_a = np.where(obs[32:38] == 1)[0][0]
    fr_o = bool(obs[38])
    wstate = obs[52:56]

    mypos_idx = self.my_pos_space.state_to_idx[my_loc]
    myact_idx = self.my_action_space.state_to_idx[my_a]
    if fr_o:
      fr_loc = np.where(obs[39:46] == 1)[0][0]
      fr_a = np.where(obs[46:52] == 1)[0][0]
      otherplayer_idx = self.otherplayer_space.state_to_idx[(fr_loc, fr_a)]
    else:
      otherplayer_idx = self.otherplayer_space.state_to_idx[None]

    wstate_idx = self.work_space.state_to_idx[tuple(wstate)]

    sidx = self.np_state_to_idx[(mypos_idx, myact_idx, otherplayer_idx,
                                 wstate_idx)]
    return sidx

  def convert_sidx_2_obs(self, sidx):
    vec_state = self.np_idx_to_state[sidx]
    mypos_idx = vec_state[0]
    mypos = self.my_pos_space.idx_to_state[mypos_idx]
    myact_idx = vec_state[1]
    myact = self.my_action_space.idx_to_state[myact_idx]
    otherplayer_idx = vec_state[2]
    otherplayer = self.otherplayer_space.idx_to_state[otherplayer_idx]
    wstate_idx = vec_state[3]
    wstate = self.work_space.idx_to_state[wstate_idx]

    my_loc = one_hot(mypos, 32)
    my_a = one_hot(myact, 6)
    if otherplayer is None:
      fr_o = np.zeros(1)
      fr_loc = np.zeros(7)
      fr_a = np.zeros(6)
    else:
      fr_o = np.ones(1)
      fr_loc = one_hot(otherplayer[0], 7)
      fr_a = one_hot(otherplayer[1], 6)

    wstate = np.array(wstate)
    obs = np.concatenate([my_loc, my_a, fr_o, fr_loc, fr_a, wstate])
    return obs


@click.command()
@click.option("--env-name", type=str, default="PO_Movers-v2")
@click.option("--seed", type=int, default=1)
def main(env_name, seed):
  print(f"Running BTIL for {env_name} with seed {seed}")

  if env_name == "PO_Movers-v2":
    converter = Converter_PO_Movers()
  elif env_name == "PO_Flood-v2":
    converter = Converter_PO_Flood()
  else:
    raise NotImplementedError()

  num_trajs = 50
  supervision = 0.2
  n_labeled = int(num_trajs * supervision)
  cur_dir = os.path.dirname(__file__)
  data_path = os.path.join(cur_dir, f"data/{env_name}_100.pkl")

  list_expert_trajs = load_trajectories(data_path, num_trajs, seed + 42)
  btil_trajs = converter.convert_trajectories(list_expert_trajs, n_labeled)

  # use btil
  batch_size = 500
  save_dir = os.path.join(cur_dir, "result/btil_models/")
  if not os.path.exists(save_dir):
    os.mkdir(save_dir)

  n_agents = 2
  for i_a in range(n_agents):
    btil_models = BTIL_SVI(btil_trajs[i_a],
                           converter.num_states, (4, ), (6, ),
                           (False, False, True),
                           500,
                           epsilon=0.1,
                           lr=1,
                           decay=0,
                           no_gem=True)
    btil_models.set_prior(gem_prior=3, tx_prior=3, pi_prior=3)
    btil_models.initialize_param()
    btil_models.do_inference(batch_size=batch_size)

    policy_file_name = env_name + "_btil_svi_policy"
    policy_file_name = os.path.join(save_dir, policy_file_name)
    np.save(policy_file_name + f"_a{i_a + 1}_s{seed}",
            btil_models.list_np_policy[0])

    tx_file_name = env_name + "_btil_svi_tx"
    tx_file_name = os.path.join(save_dir, tx_file_name)
    np.save(tx_file_name + f"_a{i_a + 1}_s{seed}", btil_models.list_Tx[0].np_Tx)

    bx_file_name = env_name + "_btil_svi_bx"
    bx_file_name = os.path.join(save_dir, bx_file_name)
    np.save(bx_file_name + f"_a{i_a + 1}_s{seed}", btil_models.list_bx[0])


if __name__ == "__main__":
  main()

# Deep Team Imitation Learner
This repository contains the implementation of **DTIL**, proposed in `Hierarchical Imitation Learning of Team Behavior from Heterogeneous Demonstrations`, AAMAS 2025.

## Installation
We have tested this implementation with **Python 3.8**. Please use a virutal environment for setup.

Clone this repository and install it with the following commands:
```
cd DTIL/
pip install -e .
pip install -r requirements.txt
```

Next, download [on-policy](https://github.com/marlbenchmark/on-policy.git) package and install it:
```
git clone https://github.com/marlbenchmark/on-policy.git
pip install -e ./on-policy
```

Finally, install StarCraft II following the instructions provided in [SMACv2](https://github.com/oxwhirl/smacv2?tab=readme-ov-file#getting-started).

Also, unzip `train/training_data.zip` into the `train/data/` directory.


## Training
To train an algorithm on a specific domain with a given supervision degree (0.0-1.0), run:
```
python train/run_algs.py alg=ALG_NAME env=ENV_NAME base=ENV_BASE supervision=SUPERVISION_DEGREE
```

To run all experiments: 
```
train/scripts/run_all.sh
```

## Citation
```
@inproceedings{seo2025hierarchical,
  title={Hierarchical Imitation Learning of Team Behavior from Heterogeneous Demonstrations},
  author={Seo, Sangwon and Unhelkar, Vaibhav},
  booktitle={Proc. of the 24th International Conference on Autonomous Agents and Multiagent Systems},
  pages={1886--1894},
  year={2025}
}
```
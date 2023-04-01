# Reinforcement Learning from Visual Inputs for Robotic Manipulation

The project aims to combine different advances in robot arm manipulation in order to better address cluttered scenes with multiple objects. The papers we try to combine in this approach are [VPG](https://github.com/andyzeng/visual-pushing-grasping), [ProDMP](https://arxiv.org/pdf/2210.01531) and [Differentiable Trust Regions](https://arxiv.org/pdf/2101.09207). The VPG (visual pushing for grasping) approach learns to grasp different objects by also leveraging pushing as an auxiliary task to improve grasping capabilities of the end-effector. The other important characteristic of VPG is that it defines a critic network in pixel space for each task (pushing and grasping). The pixel with the higher expected reward given from the critic is then chosen as the target pixel where the next action is carried out. In VPG the actions for grasp and push are preemptively defined and carried out through collision-free IK (inverse kinematics). For our extension to VPG, we instead employ a hierarchical model where the local decision (after the VPG framework has chosen the target pixel) is carried out through movement primitives. Both decisions in the global scale (pixel level) and the local scale (trajectory carried out through motion primitives) are optimized by reinforcement learning (RL). In hopes to strengthen the decision making procedure, we consider using differentiable trust region (TR) layers which offer a more theoretical approach to constraining TR compared to the approximations used in [TRPO](https://arxiv.org/pdf/1502.05477) and [PPO](https://arxiv.org/pdf/1707.06347).

## Setup

The implementation has been tested with python 3.8.10 on a Ubuntu 20.04.5 LST machine. To setup the environment install the dependencies listed in `requirements.txt`. Setup example using an [Anaconda](https://www.anaconda.com/products/distribution) environment:
```
# after installing Anaconda
# create environment, use any python version you want
conda create -n vpg_dmp python=3.8.10

# activate environment
conda activate vpg_dmp

# install requirements
pip install requirements.txt

# Beware, some of the dependencies reference github repos that might be private
```

## Running the algorithm

To run the algorithm the `main.py` script serves as entry point and it takes one required arguments with the flag `-p` or `--path` which specifies a `yaml` configuration file (check `configs/` directory for examples) that is used to define the settings of the algorithm. An optional parameter is `-d` or `--dir` which specifies the directory name under the `experiments/` directory  where the current algorithm run will be saved, if the parameter is not defined the directory name is automatically generated based on the configuration file and time.
```
# example of a run
python src/main.py -p configs/base_config.yaml -d temp
```

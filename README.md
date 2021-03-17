# gym-traffic

OpenAI Gym Environment for traffic lights simulation:

* Street network as directed graph
    * The world consists of multiple intersections.
    * Intersections are connected with streets.
    * Intersections have traffic lights for each incoming street.
    * Traffic lights can either be red or green.
    * For each intersection there cannot be more than one green light.
    * Vehicles can spawn at some intersections with a predefined route.
    * Vehicles drive on streets and stop at red traffic lights.
    * Vehicles do not crash into each other.
    
An Agent has to control the traffic lights.

Example using the `graph_3x3circle`-world and the `ppo-meta-acceleration`-agent:
![](src/examples/video.gif)

## Requirements

This project was solved using Python 3.8.6

Requires:

* OpenAI gym (`pip intall gym`)
* PyTorch ([Installation Guide](https://pytorch.org/get-started/locally/))
* OpenCV (`pip install opencv-python`) (for visualization)
* tqdm (`pip install tqdm`) (for progressbar in evaluation)
* stable-baselines3 (for training agents) (`pip install stable-baselines3`)

## Overview
There are different environment and reward types which can be combined in any way:

### Environment types
There are two different environments from which one can choose: 

#### Multi-Discrete environment (Conventional approach)
In this environment all traffic lights have to be controlled simultaneously.

The observationspace varies depending on the design of the used street network.

The actionspace can be described as Multi-Discrete, as every intersection has its own discrete action.


From this following features derive:
| Features          |       |    Features          |       |        
| :---              | :---: |         ---:         | :---: |
| Fully observable  |  YES  | Partially observable |    NO |
| Static            |   NO  | Dynamic              |   YES |
| Discrete          |   NO  | Continuous           |   YES |
| Deterministic     |   NO  | Stochastic           |   YES |
| Single agent      |  YES  | Multi-agent          |    NO |
| Competitive       |   NO  | Collaborative        |   YES |
| Episodic          |  YES  | Sequential           |    NO |

#### Single-Discrete environment (Generalized approach)
In this environment only on intersection can be controlled at one timestep and observation is only given for this intersection.
It is assumed that there are k incoming streets at every intersection.

This results in a observationspace of k values, independent of the intersection or design of the street network.

The actionspace is a single discrete action ranging from 0 to k for each timestep.

Therefore a slightly different feature-matrix derives:
| Features          |       |    Features          |       |        
| :---              | :---: |         ---:         | :---: |
| Fully observable  |   NO  | Partially observable |   YES |
| Static            |   NO  | Dynamic              |   YES |
| Discrete          |   NO  | Continuous           |   YES |
| Deterministic     |   NO  | Stochastic           |   YES |
| Single agent      |  YES  | Multi-agent          |    NO |
| Competitive       |   NO  | Collaborative        |   YES |
| Episodic          |  YES  | Sequential           |    NO |

### Rewardfunction
#### mean velocity
With this reward type the mean velocity for all vehicles is calculated and normalized around approximately 0: r=(r'-5)/5

#### mean acceleration
With this reward type the acceleration is calculated by dividing the difference of two mean velocities by dt.

## Hyperparameter
#### Conventional approach
| parameter | description | possible settings/default | 
| --------- | ----------- | ----------------- |
| horizon | number of steps in until done | 1000 |
| calculation_frequency | time steps in which the simulation is calculated | 0.01 |
| action_frequency | time which has to pass until env asks for new action | 1 |
| reward_type | Method that gives the reward | `mean_velocity`, `acceleration` |

#### Generalized approach
| parameter | description | possible settings/default | 
| --------- | ----------- | ----------------- |
| world | The actual design of the street network |
| horizon | number of steps in until done | 1000 |
| calculation_frequency | time steps in which the simulation is calculated | 0.01 |
| action_frequency | time which has to pass until env asks for new action | 1 |
| reward_type | Method that gives the reward | `mean_velocity`, `acceleration` |
| shuffle_streets | order of observation is randomized if set to true. Can be helpful for training |`True`, `False` |

## Hardware Setup
The training was done with following setup:

|  |  |
| ---- | ---- |
OS | Windows 10, Version 1909
CPU | Intel(R) Core(TM) i5-10500 CPU @ 3.10GHz, 3096 MHz
RAM | 32GB DDR4 (2666MHz)
GPU | NVIDIA Quadro P2200
## Training
An example for training an agent with this environment is given in `src/examples/train.py`
Execute with:
````shell
cd src
python examples/train.py
````


## Results
Each model was evaluated at least 5 times.

### Conventional approach
| Algorithm | Mean Velocity | Sum reward/1000 steps (acceleration) | Steps/Hours trained | 
| --------- | ------------- | ------------------------------------ | ------------- |
random | 4.200 | 2.626 | - 
ppo-acceleration | 6.120  | 4.672 | 1.5M steps (~10 hours)

### Generalized approach
| Algorithm | Mean Velocity | Sum reward/1000 steps (acceleration) | Steps/Hours trained | 
| --------- | ------------- | ------------------------------------ | ------------- |
random | 5.525 | 5.645 | - 
argmax | 8.115 | 8.023 | - 
PPO-velocity | 6.383 | 5.815 | 1.5M steps (~9h) 
PPO-velocity-shuffled | 7.736 | 8.548 | 1.25M steps (~ 7.5h)
PPO-acceleration-shuffled-1 | 7.991 | 7.522 | 1.25M steps (~ 7.5h)
PPO-acceleration-shuffled-2 | 7.987 | 5.775 | 1.25M steps (~ 7.5h)
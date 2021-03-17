# gym-traffic

OpenAI Gym Environment for traffic lights simulation

## Installation

### Requirements

This project was solved using Python 3.8.6

Requires:

* OpenAI gym (`pip intall gym`)
* PyTorch ([Installation Guide](https://pytorch.org/get-started/locally/))
* OpenCV (`pip install opencv-python`) (for visualization)
* tqdm (`pip install tqdm`) (for progressbar in evaluation)
* stable-baselines3 (for training agents) (`pip install stable-baselines3`)

## Features

### Hyperparameter
First of all there are two possible environments.

#### Multi-Discrete environment
| parameter | description | possible settings/default | 
| --------- | ----------- | ----------------- |
| horizon | number of steps in until done | 1000 |
| reward_type | Method that gives the reward | `mean_velocity`, `acceleration` |
|

#### Single-Discrete environment
| parameter | description | possible settings/default | 
| --------- | ----------- | ----------------- |
| horizon | number of steps in until done | 1000 |
| reward_type | Method that gives the reward | `mean_velocity`, `acceleration` |
| shuffle_streets | order of observation is randomized if set to true. Can be helpful for training |`True`, `False` |
|  |  |

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

### Single Traffic light environment

| Algorithm | Hours trained | Mean Velocity | Mean reward (acceleration) |
| --------- | ------------- | ------------- | -------------------------- |
| random | - | 
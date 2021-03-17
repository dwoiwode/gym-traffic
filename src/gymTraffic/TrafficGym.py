import time
from pathlib import Path

import gym
import numpy as np
import tqdm
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import VecEnv, SubprocVecEnv, VecFrameStack

from worlds.graph.trafficlights import TrafficLight
from worlds.savepoints import graph_3x3circle
from worlds.world import World
from stable_baselines3 import PPO, HER, SAC, DQN, A2C
import logging


class TrafficGym(gym.Env):
    reward_types = ["mean_velocity", "acceleration"]
    def __init__(self, build_world_function=graph_3x3circle, action_frequency=1, calculation_frequency=0.01, horizon=1000,
                 reward_type="mean_velocity"):
        self.world = build_world_function()
        self.action_frequency = action_frequency
        self.calculation_frequency = calculation_frequency
        self.horizon = horizon
        self._next_action = 0
        self.logger = logging.getLogger("TrafficGym")
        self.reward_type = reward_type
        self._old_mean_velocity = 0  # Needed for reward calculation if reward_type == "acceleration"

        assert isinstance(self.world, World)

        n_traffic_lanes = [len(crossing.incoming) for crossing in self.world.waypoints if
                           isinstance(crossing, TrafficLight)]
        self.action_space = gym.spaces.MultiDiscrete(n_traffic_lanes)
        # Observation per Crossing:
        # * Observation per lane
        #   * Nearest car
        low = []
        high = []
        for tl in self.world.traffic_light_waypoints:
            low += list(np.ones_like(self._get_observation_for(tl)) * -1)
            high += list(np.ones_like(self._get_observation_for(tl)) * 2)

        low = np.asarray(low, dtype=np.float32)
        high = np.asarray(high, dtype=np.float32)
        self.observation_space = gym.spaces.Box(low, high)
        self.reward_range = (-2, 2)

    def reset(self):
        self.world.reset()
        self._next_action = self.world.t
        return self.get_observation()

    def render(self, mode='human'):
        self.world.render()

    def step(self, action):
        self.logger.debug(f"step<{self.world.t:.2f}>({action})")
        for state, crossing in zip(action, self.world.traffic_light_waypoints):
            assert isinstance(crossing, TrafficLight)
            if state < len(crossing.incoming):
                crossing.green = [crossing.incoming[state]]

        # Simulate until next action is required
        self._next_action += self.action_frequency
        while self.world.t < self._next_action:
            self.world.step(self.calculation_frequency)
        obs = self.get_observation()
        reward = self.get_reward(self.action_frequency)
        return obs,reward, self.horizon <= self.world.t, {}

    def get_observation(self):
        obs = []
        for tl in self.world.traffic_light_waypoints:
            obs += self._get_observation_for(tl)
        return np.asarray(obs)

    def _get_observation_for(self, traffic_light: TrafficLight):
        obs = []
        for street in traffic_light.incoming:
            if len(street.vehicles) == 0:
                obs.append(-1)
            else:
                closest_car = sum(street.vehicles.values())
                obs.append(closest_car / street.length())
            # obs.append(len(street.vehicles) *10 / street.length())
        return obs

    def get_reward(self, dt=1.):
        mean_velocity = self.world.mean_velocity
        if self.reward_type == "mean_velocity":
            return (mean_velocity - 5) / 5
        if self.reward_type == "acceleration":
            acceleration = (mean_velocity - (self._old_mean_velocity or 0)) / dt
            self._old_mean_velocity = mean_velocity
            return acceleration


class TrafficGymMeta(TrafficGym):
    def __init__(self, build_world_function=graph_3x3circle, action_frequency=1,
                 calculation_frequency=0.01, horizon=1000, reward_type="mean_velocity",
                 shuffle_streets=True):
        super().__init__(build_world_function=build_world_function, action_frequency=action_frequency,
                         calculation_frequency=calculation_frequency, horizon=horizon,reward_type=reward_type)
        self._traffic_light_counter = 0
        self._observation_order = np.argsort(np.random.random(8))
        self.shuffle_streets = shuffle_streets
        self.action_array = self.action_space.sample()

        self._original_action_space = self.action_space

        self.action_space = gym.spaces.Discrete(8)
        low = np.zeros(8, dtype=np.float32)
        high = np.ones(8, dtype=np.float32) * 2
        self.observation_space = gym.spaces.Box(low, high)

    def get_observation(self):
        observation_part = self._get_observation_for(self.world.traffic_light_waypoints[self._traffic_light_counter])
        observation = np.ones(8, dtype=np.float32) * -1
        observation[:len(observation_part)] = observation_part
        if self.shuffle_streets:
            return observation[self._observation_order]
        return observation

    def step(self, action):
        # Single action -> action array
        if self.shuffle_streets:
            self.action_array[self._traffic_light_counter] = self._observation_order[action]
        else:
            self.action_array[self._traffic_light_counter] = action

        # Prepare for next observation
        self._traffic_light_counter = (self._traffic_light_counter + 1) % len(self.world.traffic_light_waypoints)
        self._observation_order = np.argsort(np.random.random(8))

        return super(TrafficGymMeta, self).step(self.action_array)

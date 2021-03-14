import time

import gym
import numpy as np
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
    def __init__(self, build_world_function=graph_3x3circle, action_frequency=1, calculation_frequency=0.01, horizon=1000):
        self.world = build_world_function()
        self.action_frequency = action_frequency
        self.calculation_frequency = calculation_frequency
        self.horizon = horizon
        self._next_action = 0
        self.logger = logging.getLogger("TrafficGym")

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
            low += list(np.zeros_like(self._get_observation_for(tl)))
            high += list(np.ones_like(self._get_observation_for(tl)))

        low = np.asarray(low, dtype=np.float32)
        high = np.asarray(high, dtype=np.float32)
        self.observation_space = gym.spaces.Box(low, high)

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
        reward = self.get_reward()
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

    def get_reward(self):
        return (self.world.mean_velocity - 5) / 5


class TrafficGymMeta(TrafficGym):
    def __init__(self, build_world_function=graph_3x3circle, action_frequency=1, calculation_frequency=0.01, horizon=1000):
        super().__init__(build_world_function=build_world_function, action_frequency=action_frequency,
                         calculation_frequency=calculation_frequency, horizon=horizon)
        self._traffic_light_counter = 0
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
        return observation

    def step(self, action):
        self.action_array[self._traffic_light_counter] = action
        self._traffic_light_counter = (self._traffic_light_counter + 1) % len(self.world.traffic_light_waypoints)
        ret = super(TrafficGymMeta, self).step(self.action_array)
        return ret


class RandomAgent:
    def __init__(self, env):
        self.env = env

    def get_action(self, observation):
        return self.env.action_space.sample()


def stable_baselines(env):
    def make_env(env, rank, seed=0):
        """
        Utility function for multiprocessed env.

        :param env_id: (str) the environment ID
        :param seed: (int) the inital seed for RNG
        :param rank: (int) index of the subprocess
        """

        def _init():
            env_ = env()
            # Important: use a different seed for each environment
            env_.seed(seed + rank)
            return env_

        return _init
    # logging.basicConfig(level=logging.NOTSET)
    # model = PPO('MlpPolicy', env,verbose=1)
    num_cpu = 10  # Number of processes to use
    env_vec = SubprocVecEnv([make_env(env, i) for i in range(num_cpu)])
    # env = Monitor(env)

    # Create the vectorized environment

    model = PPO(
        "MlpPolicy",
        env_vec,
        tensorboard_log="./ppo_trafficgym_tensorboard/",
        verbose=2,
        learning_rate=1e-2,
        # gamma=0.95,
        batch_size=256,
        # batch_size=512,
        # policy_kwargs=dict(net_arch=[256, 512, 256]),
        policy_kwargs=dict(net_arch=[64, 64]),
    )
    # model.load("ppo.stable_baselines")
    env = Monitor(env())
    print("Evaluating...")
    evaluation = evaluate_policy(model, env)
    print("Eval1:", evaluation)
    t1 = time.time()
    model.learn(2000)
    t2 = time.time()
    print(f"Learning took {t2-t1} seconds")
    model.save("ppo.stable_baselines")
    print("Evaluating...")
    evaluation = evaluate_policy(model, env)
    print("Eval2:", evaluation)

def test_baseline(env, render=True):
    env = env()
    model = PPO(
        "MlpPolicy",
        env,
        tensorboard_log="./ppo_trafficgym_tensorboard/",
        verbose=2,
        learning_rate=1e-2,
        # gamma=0.95,
        batch_size=256,
        # batch_size=512,
        # policy_kwargs=dict(net_arch=[256, 512, 256]),
        policy_kwargs=dict(net_arch=[64, 64]),
    )
    model.load("ppo.stable_baselines")

    done = False
    observation = env.reset()
    t = []
    velocities = []
    rewards = []
    while not done:
        action, _ = model.predict(observation)
        # print(action)
        observation, reward, done, info = env.step(action)
        print(observation)
        print(reward)
        if render:
            env.render()

        velocities.append(env.world.mean_velocity)
        t.append(env.world.t)
        rewards.append(reward)

    import matplotlib.pyplot as plt

    plt.plot(t,velocities,".",label="Velocity")
    plt.tight_layout()
    plt.show()
    plt.plot(t,rewards,".",label="Reward")
    plt.tight_layout()
    plt.show()


def custom_run(env):
    env = env()
    agent = RandomAgent(env)

    for i in range(2):
        observation = env.reset()
        done = False
        while not done:
            action = agent.get_action(observation)
            # print("Action:", action)
            observation, reward, done, info = env.step(action)
            # print("Obs:",observation)
            print("Reward:",reward)
            env.render()


if __name__ == '__main__':
    from worlds.savepoints import *

    env = lambda: TrafficGymMeta(graph_3x3circle, horizon=1000)

    stable_baselines(env)
    test_baseline(env, render=False)
    # custom_run(env)

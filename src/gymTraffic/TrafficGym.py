import gym
import numpy as np
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import VecEnv

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
            crossing.green = [crossing.incoming[state]]

        # Simulate until next action is required
        self._next_action += self.action_frequency
        while self.world.t < self._next_action:
            self.world.step(self.calculation_frequency)
        obs = self.get_observation()
        reward = self.get_reward()
        return obs,reward, self.horizon <= self.world.t, {}

    def get_observation(self):
        return np.asarray([self._get_observation_for(tl) for tl in self.world.traffic_light_waypoints]).reshape(-1)

    def _get_observation_for(self, traffic_light: TrafficLight):
        obs = []
        for street in traffic_light.incoming:
            if len(street.vehicles) == 0:
                obs.append(0)
            else:
                obs.append(max(street.vehicles.values()) / street.length())
            # obs.append(len(street.vehicles) *10 / street.length())
        return obs

    def get_reward(self):
        return (self.world.mean_velocity - 5) / 5


class RandomAgent:
    def __init__(self, env):
        self.env = env

    def get_action(self, observation):
        return self.env.action_space.sample()


def stable_baselines(env):
    # logging.basicConfig(level=logging.NOTSET)
    # model = PPO('MlpPolicy', env,verbose=1)
    env = Monitor(env)
    # venv = make_vec_env(lambda: env,10)
    model = PPO(
        "MlpPolicy",
        env,
        tensorboard_log="./ppo_trafficgym_tensorboard/",
        verbose=2,
        learning_rate=1e-2,
        # gamma=0.95,
        # batch_size=256,
        batch_size=512,
        policy_kwargs=dict(net_arch=[256, 256, 256]),
    )
    evaluation = evaluate_policy(model, env)
    print(evaluation)
    model.learn(1)
    model.save("ppo.stable_baselines")
    evaluation = evaluate_policy(model, env)
    print(evaluation)

    obs = env.reset()
    for i in range(1000):
        action, _states = model.predict(obs)
        obs, rewards, dones, info = env.step(action)
        env.render()


def custom_run(env):
    agent = RandomAgent(env)

    for i in range(2):
        observation = env.reset()
        done = False
        while not done:
            action = agent.get_action(observation)
            # print(action)
            observation, reward, done, info = env.step(action)
            print(observation)
            print(reward)
            env.render()


if __name__ == '__main__':
    from worlds.savepoints import *

    env = TrafficGym(graph_cross,horizon=100)

    stable_baselines(env)
    # custom_run(env)

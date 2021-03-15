import time

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


class RandomAgent:
    def __init__(self, env):
        self.env = env

    def get_action(self, observation):
        return self.env.action_space.sample()


def stable_baselines(env, name="model"):
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
    # model = model.load("PPO_meta_fixedorder_1_07.stable_baselines")
    env = Monitor(env())
    print("Evaluating...")
    evaluation = evaluate_policy(model, env)
    print("Eval1:", evaluation)
    t1 = time.time()
    for i in range(8):  # Each iteration takes about a hour to complete
        try:
            model.learn(1000 * 125)
            print(f"Save model {i}")
            model.save(f"{name}{i:02d}.stable_baselines")
        except KeyboardInterrupt:
            print("Interrupted by KeyBoard")
            break
    t2 = time.time()
    print(f"Learning took {t2-t1} seconds")
    # model.save("ppo.stable_baselines")
    print("Evaluating...")
    evaluation = evaluate_policy(model, env)
    print("Eval2:", evaluation)

def test_baseline(env, savepoint="random",render=True):
    env = env()
    print()
    print("="*100)
    print("="*5,f"Savepoint: {savepoint}, reward_type: {env.reward_type}", "="*5)
    print("="*100)
    # env.seed(42)
    # model = PPO(
    #     "MlpPolicy",
    #     env,
    #     tensorboard_log="./ppo_trafficgym_tensorboard/",
    #     verbose=2,
    #     learning_rate=1e-2,
    #     # gamma=0.95,
    #     batch_size=256,
    #     # batch_size=512,
    #     # policy_kwargs=dict(net_arch=[256, 512, 256]),
    #     policy_kwargs=dict(net_arch=[64, 64]),
    # )
    if savepoint not in ["random", "argmax", None]:
        model = PPO.load(savepoint)

    # evaluation = evaluate_policy(model, env)

    done = False
    observation = env.reset()
    t = []
    velocities = []
    rewards = []
    actions = []
    progress = tqdm.tqdm(total=env.horizon)
    while not done:
        progress.update(1)
        if savepoint == "random":
            action = env.action_space.sample()
        elif savepoint == "argmax":
            action = np.argmax(observation)
        else:
            action, _ = model.predict(observation)

        # print(action)
        actions.append(action)
        observation, reward, done, info = env.step(action)
        # print(observation)
        # print(reward)
        if render:
            env.render()

        velocities.append(env.world.mean_velocity)
        t.append(env.world.t)
        rewards.append(reward)

    progress.close()
    import matplotlib.pyplot as plt

    print("Actions:",np.unique(actions,return_counts=True))

    plt.plot(t,velocities,".",label="Velocity")
    plt.plot(t,rewards,".",label=f"Reward ({env.reward_type})")
    plt.legend()
    plt.title(f"{savepoint}")
    plt.tight_layout()
    plt.savefig(f"figures/{savepoint}.png")
    # plt.show()

    print("Mean vel:",np.mean(velocities))
    print("Std vel:",np.std(velocities))
    print("Sum reward:", np.sum(rewards))
    return np.mean(velocities), np.std(velocities), np.mean(rewards), np.std(rewards), np.sum(rewards)


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


def results_evaluation(savepoint_folder=""):
    from pathlib import Path

    resultFile = Path("results.csv")
    if not resultFile.exists():
        with resultFile.open("w") as d:
            d.write("filename,vel_mean,vel_std,rew_mean,rew_std,rew_sum\n")
    for savepoint in Path(savepoint_folder).iterdir():
        if "ppo" in savepoint.name.lower() and "meta" in savepoint.name.lower():
            values = test_baseline(env, str(savepoint), render=False)
            with resultFile.open("a") as d:
                d.write(",".join(map(str, [savepoint] + list(values))) + "\n")
            print(savepoint)


if __name__ == '__main__':
    from worlds.savepoints import *

    env = lambda: TrafficGymMeta(graph_3x3circle, horizon=1000,reward_type="mean_velocity")

    stable_baselines(env,name="PPO_meta_fixedorder_1_")
    # stable_baselines(env,name="PPO_meta_fixedorder_2_")
    # test_baseline(env,render=False,savepoint="argmax")
    test_baseline(env, savepoint="PPO_meta_fixedorder_shuffle_1_10.stable_baselines",render=True)
    # test_baseline(env, savepoint="PPO_meta_fixedorder_2_07.stable_baselines",render=False)
    # custom_run(env)

    results_evaluation("savepoints/all")


import time
from typing import Optional

from gym.wrappers import Monitor
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import SubprocVecEnv


def train(env_function, name="model", n_processes: int = 6, seed: int = 0, load_checkpoint: Optional[str] = None,
          from_index=0, to_index=12, steps_per_episode=125 * 1000):
    """
    Trains a model with a given environment

    :param env_function: Function that creates an gym.Env
    :param name: name for saving
    :param n_processes: number of processes used for training
    :param seed:
    :param load_checkpoint: if None: Create new model. Else: Load model from file
    :param steps_per_episode: Number of steps for model.learn()
    :param from_index: starting with this episode (for continuing training later than 0)
    :param to_index: last index of episode
    :return:
    """

    def make_env(rank: int):
        """
        Utility function for multiprocessed env.

        :param rank: index of the subprocess (needed to update seed)
        """

        def _init():
            env = env_function()
            # Important: use a different seed for each environment
            env.seed(seed + rank)
            return env

        return _init

    # Create the vectorized environment
    env_vector = SubprocVecEnv([make_env(i) for i in range(n_processes)])

    # Create model
    if load_checkpoint is None:
        model = PPO(
            "MlpPolicy",
            env_vector,
            tensorboard_log="./ppo_trafficgym_tensorboard/",
            verbose=2,
            learning_rate=1e-2,
            # gamma=0.95,
            batch_size=256,
            policy_kwargs=dict(net_arch=[64, 64]),
        )
    else:
        model = PPO.load(load_checkpoint)

    # Evaluate before training
    env = Monitor(env_function())
    print("Evaluating...")
    evaluation = evaluate_policy(model, env)
    print("Eval1:", evaluation)

    # Actual training
    t1 = time.time()
    for i in range(from_index, to_index+1):
        try:
            model.learn(steps_per_episode)
            print(f"Save model {i}")
            model.save(f"{name}{i:02d}.stable_baselines")
        except KeyboardInterrupt:
            print("Interrupted by KeyBoard")
            break
    t2 = time.time()
    print(f"Learning took {t2 - t1} seconds")

    # Evaluate after training
    print("Evaluating...")
    evaluation = evaluate_policy(model, env)
    print("Eval2:", evaluation)


if __name__ == '__main__':
    from gymTraffic.TrafficGym import TrafficGym, TrafficGymMeta
    import worlds.savepoints as save

    # env_function = lambda: TrafficGym(save.graph_3x3circle, horizon=1000, reward_type="acceleration")
    env_function = lambda: TrafficGymMeta(save.graph_3x3circle, horizon=1000, reward_type="acceleration")

    train(env_function, "ppo_meta")

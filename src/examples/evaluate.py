from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
import logging
import tqdm


# logging.basicConfig(level=logging.INFO)

def evaluate_model(env, savepoint="random", render=True, iteration=0, plot_results="show"):
    """
    Runs an evaluation on an environment given a model.
    The savepoint can either be `argmax`, `random` or a path to a savepoint for a PPO-Algorithm
    plot_results can either be  `show` or `save`
    """
    logger = logging.getLogger(f"Eval[{savepoint}:{iteration}]")
    logger.info("")
    logger.info("=" * 120)
    logger.info(f"{f' Savepoint: {savepoint}, reward_type: {env.reward_type} ':=^120}")
    logger.info("=" * 120)

    if savepoint not in ["random", "argmax", None]:
        model = PPO.load(savepoint)

    # Evaluation
    t = []
    velocities = []
    rewards = []
    actions = []

    observation = env.reset()
    progress = tqdm.tqdm(total=env.horizon)
    done = False
    while not done:
        progress.update(1)
        # Get Action
        if savepoint == "random":
            action = env.action_space.sample()
        elif savepoint == "argmax":
            action = np.argmax(observation)
        else:
            action, _ = model.predict(observation)

        actions.append(action)
        observation, reward, done, info = env.step(action)
        logger.debug(f"Action: {action}")
        logger.debug(f"Observation: {observation}")
        logger.debug(f"Reward: {reward}")
        if render:
            env.render()

        velocities.append(env.world.mean_velocity)
        t.append(env.world.t)
        rewards.append(reward)

    progress.close()

    # Plotting diagrams
    if plot_results is not None:
        plt.plot(t, velocities, ".", label="Velocity")
        plt.plot(t, rewards, ".", label=f"Reward ({env.reward_type})")
        plt.ylim(-3, 16)
        plt.legend()
        plt.title(f"{savepoint}")
        plt.tight_layout()
        if plot_results == "show":
            plt.show()
        elif plot_results == "save":
            plt.savefig(f"figures/{Path(savepoint).name}_{iteration}.png")
        plt.close()

    logger.info("Actions:", np.unique(actions, return_counts=True))
    logger.info("Mean vel:", np.mean(velocities))
    logger.info("Std vel:", np.std(velocities))
    logger.info("Sum reward:", np.sum(rewards))
    return np.mean(velocities), np.std(velocities), np.mean(rewards), np.std(rewards), np.sum(rewards)


def evaluate_folder(savepoint_folder:Path=Path(""), result_file:Path=Path("results.csv"), iteration=0):
    """
    Evaluates a whole range of savepoints in a folder
    :param savepoint_folder: path to folder
    :param result_file: path to file for results. will be created if it does not exist
    :param iteration:
    :return:
    """
    # Check whether file exists and create otherwise
    if not result_file.exists():
        result_file.parent.mkdir(exist_ok=True, parents=True)
        with result_file.open("w") as d:
            d.write("filename,vel_mean,vel_std,rew_mean,rew_std,rew_sum,iteration\n")

    # Iterate over all checkpoints
    for savepoint in savepoint_folder.iterdir():
        values = evaluate_model(env, str(savepoint), render=False, iteration=iteration)
        with result_file.open("a") as d:
            d.write(",".join(map(str, [savepoint] + list(values) + [iteration])) + "\n")


if __name__ == '__main__':
    from gymTraffic.TrafficGym import TrafficGym, TrafficGymMeta
    import worlds.savepoints as save

    # env = TrafficGym(save.graph_3x3circle, horizon=1000, reward_type="acceleration", action_frequency=1)
    env = TrafficGymMeta(save.graph_3x3bidirectional, horizon=1000, reward_type="acceleration", action_frequency=1)

    evaluate_model(env, "../../savepoints/ppo_meta_acceleration_1_10.stable_baselines")
    evaluate_model(env, "random")

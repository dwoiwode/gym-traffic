from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
import logging
import tqdm

# logging.basicConfig(level=logging.INFO)

def evaluate_model(env, savepoint="random", render=True, iteration=0, plot_results="show"):
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

    print("Actions:", np.unique(actions, return_counts=True))
    print("Mean vel:", np.mean(velocities))
    print("Std vel:", np.std(velocities))
    print("Sum reward:", np.sum(rewards))
    return np.mean(velocities), np.std(velocities), np.mean(rewards), np.std(rewards), np.sum(rewards)


def evaluate_folder(savepoint_folder="", iteration=0):
    resultFile = Path("results_conventional.csv")
    if not resultFile.exists():
        with resultFile.open("w") as d:
            d.write("filename,vel_mean,vel_std,rew_mean,rew_std,rew_sum,iteration\n")
    for savepoint in Path(savepoint_folder).iterdir():
        values = evaluate_model(env, str(savepoint), render=False, iteration=iteration)
        with resultFile.open("a") as d:
            d.write(",".join(map(str, [savepoint] + list(values) + [iteration])) + "\n")
        print(savepoint)


if __name__ == '__main__':
    from gymTraffic.TrafficGym import TrafficGym, TrafficGymMeta
    import worlds.savepoints as save

    # env = TrafficGym(save.graph_3x3circle, horizon=1000, reward_type="acceleration", action_frequency=1)
    env = TrafficGymMeta(save.graph_3x3bidirectional, horizon=1000, reward_type="acceleration", action_frequency=1)

    evaluate_model(env, "../../savepoints/ppo_meta_acceleration_1_10.stable_baselines")

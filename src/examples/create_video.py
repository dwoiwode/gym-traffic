import cv2
import numpy as np
import tqdm
from stable_baselines3 import PPO

from gymTraffic.TrafficGym import TrafficGymMeta


def create_video(env, savepoint="random", out_filename="video.mp4", video_size=(1230, 900)):
    if savepoint not in ["random", "argmax", None]:
        model = PPO.load(savepoint)

    # Videosettings
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(out_filename, fourcc, 10, video_size)

    observation = env.reset()
    progress = tqdm.tqdm(total=env.horizon)
    done = False
    while not done:
        progress.update(env.action_frequency)
        # Get Action
        if savepoint == "random":
            action = env.action_space.sample()
        elif savepoint =="argmax":
            action = np.argmax(observation)
        else:
            action, _ = model.predict(observation)

        observation, reward, done, info = env.step(action)
        img = env.render("rgb_array")
        # cv2.imshow("Wold",img)
        # cv2.waitKey(10)

        # resized_img = cv2.resize(img, video_size, cv2.INTER_NEAREST)
        # out.write(resized_img / 255)
        # out.write(np.asarray(img, dtype=np.float64)/255)
        out.write(np.asarray(img*255, dtype=np.uint8))

    out.release()
    progress.close()
    return out


if __name__ == '__main__':
    env = TrafficGymMeta(horizon=300)
    create_video(env,"argmax","video.mp4")
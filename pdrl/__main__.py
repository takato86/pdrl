import argparse
import logging
import json
import numpy as np
import gym
import gym_m2s
from mlflow import log_artifact
from pdrl.torch.ddpg.train import train
from pdrl.torch.ddpg.optimize import optimize_hyparams

gym_m2s
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()


def preprocess(obs, r, d, info):
    g = obs["desired_goal"]
    ag = obs["achieved_goal"]
    diff_g = g - ag
    observation = obs["observation"]
    obs = np.hstack([observation, diff_g])
    return obs, r, d, info


def main():
    env_name = configs["env"]

    def create_env():
        return gym.make(env_name)

    env_fn = create_env

    if args.optimize:
        optimize_hyparams(env_fn, preprocess, configs)
    else:
        train(env_fn, preprocess, configs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--optimize", "-o", action='store_true')
    args = parser.parse_args()
    with open("config.json", "r") as f:
        configs = json.load(f)
        log_artifact("config.json")
    main()

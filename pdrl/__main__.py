import argparse
import logging
import json
import numpy as np
import gym
import gym_m2s
from pdrl.torch.ddpg.train import train
from pdrl.torch.ddpg.optimize import optimize_hyparams

gym_m2s
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()


def preprocess(obs, r, d, info):
    g = obs["desired_goal"]
    observation = obs["observation"]
    obs = np.hstack([observation, g])
    obs = obs.reshape([1, -1])
    return obs, r, d, info


def main():
    env_name = configs["env_id"]
    env_params = configs["env_params"]

    def create_env():
        env = gym.make(env_name, **env_params)
        env.seed(configs["env_seed"])
        return env

    env_fn = create_env

    if args.optimize:
        optimize_hyparams(env_fn, preprocess, configs)
    else:
        train(env_fn, preprocess, configs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--optimize", "-o", action='store_true')
    parser.add_argument("--config", "-c", type=str, default="configs/config.json")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        configs = json.load(f)
        # log_artifact("config.json")

    main()

import argparse
import logging
import gym
import gym_m2s
import torch
import numpy as np
from pdrl.torch.ddpg_rnd.train import train
from pdrl.torch.ddpg.optimize import optimize_hyparams
from pdrl.utils.config import load_config

gym_m2s
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()


def main():
    env_name = configs["env_id"]
    env_params = configs["env_params"]
    # Fix seed
    seed = configs["seed"]
    torch.manual_seed(seed)
    np.random.seed(seed)

    def create_env():
        env = gym.make(env_name, **env_params)
        env.seed(seed)
        return env

    env_fn = create_env

    if args.optimize:
        optimize_hyparams(env_fn, configs)
    else:
        train(env_fn, configs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--optimize", "-o", action='store_true')
    parser.add_argument("--config", "-c", type=str, default="configs/config.json")
    args = parser.parse_args()
    configs = load_config(args.config)
    main()

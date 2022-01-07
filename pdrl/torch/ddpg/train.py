import logging
import os
from datetime import datetime
from pdrl.torch.ddpg.learn import learn


logger = logging.getLogger()


def train(env_fn, preprocess, configs):
    n_runs = int(configs["training_params"]["nruns"])
    params = {
        "env_fn": env_fn,
        "preprocess": preprocess,
        "epochs": int(configs["training_params"]["epochs"]),
        "steps_per_epoch": int(configs["training_params"]["steps_per_epoch"]),
        "start_steps": int(configs["training_params"]["start_steps"]),
        "update_after": int(configs["training_params"]["update_after"]),
        "update_every": int(configs["training_params"]["update_every"]),
        "num_test_episodes": int(configs["eval_params"]["num_test_episodes"]),
        "max_ep_len": int(configs["training_params"]["max_ep_len"]),
        "gamma": float(configs["agent_params"]["gamma"]),
        "epsilon": float(configs["agent_params"]["epsilon"]),
        "actor_lr": float(configs["agent_params"]["actor_lr"]),
        "critic_lr": float(configs["agent_params"]["critic_lr"]),
        "replay_size": int(configs["agent_params"]["replay_size"]),
        "polyak": float(configs["agent_params"]["polyak"]),
        "l2_action": float(configs["agent_params"]["l2_action"]),
        "noise_scale": float(configs["agent_params"]["noise_scale"]),
        "batch_size": int(configs["agent_params"]["batch_size"]),
        "norm_clip": float(configs["agent_params"]["norm_clip"]),
        "norm_eps": float(configs["agent_params"]["norm_eps"])
    }

    dir_path = "runs/train"
    start_at = datetime.now()
    for t in range(n_runs):
        logger.info("START Trial {}".format(t))
        dir_name = str(start_at) + "_" + str(t)
        params["logdir"] = os.path.join(dir_path, dir_name)
        learn(**params)
        logger.info("END Trial {}".format(t))

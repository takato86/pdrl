import logging
import os
from datetime import datetime
from pdrl.experiments.pick_and_place.pipeline import create_pipeline
from pdrl.experiments.pick_and_place.pipeline import create_test_pipeline
from pdrl.torch.ddpg.learn import learn
from pdrl.utils.config import export_config
from pdrl.utils.mpi import proc_id
from pdrl.utils.file_handler import prep_dir


logger = logging.getLogger()


def train(env_fn, configs):
    n_runs = int(configs["training_params"]["nruns"])
    pipeline = create_pipeline(configs)
    test_pipeline = create_test_pipeline(configs)
    params = {
        "env_fn": env_fn,
        "pipeline": pipeline,
        "test_pipeline": test_pipeline,
        "epochs": configs["training_params"]["epochs"],
        "steps_per_epoch": configs["training_params"]["steps_per_epoch"],
        "start_steps": configs["training_params"]["start_steps"],
        "update_after": configs["training_params"]["update_after"],
        "update_every": configs["training_params"]["update_every"],
        "num_test_episodes": configs["eval_params"]["num_test_episodes"],
        "max_ep_len": configs["training_params"]["max_ep_len"],
        "gamma": configs["agent_params"]["gamma"],
        "epsilon": configs["agent_params"]["epsilon"],
        "actor_lr": configs["agent_params"]["actor_lr"],
        "critic_lr": configs["agent_params"]["critic_lr"],
        "replay_size": configs["agent_params"]["replay_size"],
        "polyak": configs["agent_params"]["polyak"],
        "l2_action": configs["agent_params"]["l2_action"],
        "noise_scale": configs["agent_params"]["noise_scale"],
        "batch_size": configs["agent_params"]["batch_size"],
        "norm_clip": configs["agent_params"]["norm_clip"],
        "norm_eps": configs["agent_params"]["norm_eps"],
        "clip_return": configs["agent_params"]["clip_return"],
        "is_pos_return": configs["agent_params"]["is_pos_return"],
        "video": True
    }

    dir_path = "runs/train"
    start_at = datetime.now()

    for t in range(n_runs):
        logger.info("START Trial {}".format(t))

        if proc_id() == 0:
            dir_name = str(start_at) + "_" + str(t)
            logdir = os.path.join(dir_path, dir_name)
            prep_dir(logdir)
            params["logdir"] = logdir
            output_cfg_path = os.path.join(logdir, "config.json")
            export_config(output_cfg_path)

        learn(**params)
        logger.info("END Trial {}".format(t))

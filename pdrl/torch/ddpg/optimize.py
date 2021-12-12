import logging
import optuna
from pdrl.torch.ddpg.learn import learn


logger = logging.getLogger(__name__)


def optimize_hyparams(env_fn, preprocess, configs):
    logger.info("OPTIMIZE HYPAPER PARAMETERS.")
    n_trials = configs["n_trials"]
    epochs = configs["epochs"]
    steps_per_epoch = configs["steps_per_epoch"]
    start_steps = configs["start_steps"]
    update_after = configs["update_after"]
    update_every = configs["update_every"]
    num_test_episodes = configs["num_test_episodes"]
    max_ep_len = configs["max_ep_len"]

    def objective(trial):
        max_gamma = 1. - 1. / max_ep_len
        min_gamma = 1. - 5. / max_ep_len
        gamma = trial.suggest_float("gamma", min_gamma, max_gamma)
        actor_lr = trial.suggest_categorical("actor_lr", [0.001, 0.0001, 0.005])
        critic_lr = trial.suggest_categorical("critic_lr", [0.001, 0.0001, 0.005])
        replay_size = trial.suggest_categorical("replay_size", [int(1E6), int(1E5), int(1E4)])
        polyak = trial.suggest_categorical("polyak", [0.95, 0.98, 0.92])
        l2_action = trial.suggest_float("l2_action", 0.5, 1.5)
        noise_scale = trial.suggest_categorical("noise_scale", [0.2, 0.3, 0.1])
        perf = learn(env_fn, preprocess, epochs, steps_per_epoch, start_steps, update_after, update_every,
                     num_test_episodes, max_ep_len, gamma, actor_lr, critic_lr, replay_size,
                     polyak, l2_action, noise_scale)
        return perf

    study = optuna.create_study()
    study.optimize(objective, n_trials=n_trials)
    best_params = study.best_params
    logger.info(best_params)
    fig = optuna.visualization.plot_optimization_history(study)
    fig.show()

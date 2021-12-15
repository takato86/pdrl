import logging
from pdrl.torch.ddpg.learn import learn


logger = logging.getLogger()


def train(env_fn, preprocess, configs):
    n_trials = int(configs["n_trials"])
    epochs = int(configs["epochs"])
    steps_per_epoch = int(configs["steps_per_epoch"])
    start_steps = int(configs["start_steps"])
    update_after = int(configs["update_after"])
    update_every = int(configs["update_every"])
    num_test_episodes = int(configs["num_test_episodes"])
    max_ep_len = int(configs["max_ep_len"])
    gamma = float(configs["gamma"])
    actor_lr = float(configs["actor_lr"])
    critic_lr = float(configs["critic_lr"])
    replay_size = int(configs["replay_size"])
    polyak = float(configs["polyak"])
    l2_action = float(configs["l2_action"])
    noise_scale = float(configs["noise_scale"])
    batch_size = int(configs["batch_size"])

    for _ in range(n_trials):
        learn(env_fn, preprocess, epochs, steps_per_epoch, start_steps, update_after, update_every,
              num_test_episodes, max_ep_len, gamma, actor_lr, critic_lr, replay_size, polyak,
              l2_action, noise_scale, batch_size)

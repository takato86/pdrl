import logging
import json
import gym
import gym_m2s
from pdrl.torch.ddpg.agent import DDPGAgent
from pdrl.torch.ddpg.replay_memory import ReplayBuffer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()


def test(test_env, agent, num_test_episodes, max_ep_len):
    for t in range(num_test_episodes):
        o, d, ep_ret, ep_len = test_env.reset(), False, 0, 0
        while(not d or (ep_len == max_ep_len)):
            o = o["observation"]
            o, r, d, _ = test_env.step(agent.act(o, 0))
            ep_ret += r
            ep_len += 1
    print(f"test return: {ep_ret}")


def main():
    env_fn = lambda: gym.make("SingleFetchPickAndPlace-v0")
    env = env_fn()
    agent = DDPGAgent(
        env.observation_space, env.action_space, configs["gamma"], configs["actor_lr"], configs["critic_lr"],
        configs["polyak"], logger)
    replay_buffer = ReplayBuffer(
        env.observation_space["observation"].shape[0], env.action_space.shape[0], configs["replay_size"])
    total_steps = configs["steps_per_epoch"] * configs["epochs"]
    o, ep_ret, ep_len = env.reset(), 0, 0

    for i in range(int(total_steps)):
        if i > configs["start_steps"]:
            a = agent.act(o["observation"], configs["noise_scale"])
        else:
            a = env.action_space.sample()

        o2, r, d, _ = env.step(a)
        replay_buffer.store(o["observation"], a, r, o2["observation"], d)
        ep_len += 1
        ep_ret += r

        if i >= configs["update_after"] and i % configs["update_every"] == 0:
            for _ in range(configs["update_every"]):
                batch = replay_buffer.sample_batch()
                agent.update(batch)

        if (i+1) % configs["steps_per_epoch"] == 0:
            epoch = (i+1) // configs["steps_per_epoch"]
            print(f"Epoch {epoch}\n-------------------------------")
            print(f"return: {ep_ret}   [{i:>7d}/{int(total_steps):>7d}]")
            test(env_fn(), agent, configs["num_test_episodes"], configs["max_ep_len"])


if __name__ == "__main__":
    with open("config.json", "r") as f:
        configs = json.load(f)

    main()

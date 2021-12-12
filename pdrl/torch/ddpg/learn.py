import logging
from statistics import mean
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from pdrl.torch.ddpg.agent import DDPGAgent
from pdrl.torch.ddpg.replay_memory import ReplayBuffer

logger = logging.getLogger(__name__)


def test(test_env, agent, preprocess, num_test_episodes, max_ep_len):
    ep_rets, ep_lens, is_successes = [], [], []
    for _ in range(num_test_episodes):
        o, d, rets, is_success, ep_len = test_env.reset(), False, [], False, 0
        logger.debug("test reset initial obs: {}".format(o))
        o, _, _, _ = preprocess(o, 0, False, None)
        while(not d or (ep_len < max_ep_len)):
            o, r, d, info = test_env.step(agent.act(o, 0))
            o, r, d, info = preprocess(o, r, d, info)
            rets.append(r)
            is_success |= bool(info["is_success"])
            ep_len += 1
        ep_rets.append(sum(rets))
        ep_lens.append(ep_len)
        is_successes.append(is_success)
    # logger.info(f"test return: {ep_ret}")
    return mean(ep_rets), mean(ep_lens), mean(is_successes)


def learn(env_fn, preprocess, epochs, steps_per_epoch, start_steps, update_after, update_every, num_test_episodes,
          max_ep_len, gamma, actor_lr, critic_lr, replay_size, polyak, l2_action, noise_scale):
    env = env_fn()
    writer = SummaryWriter()
    total_steps = steps_per_epoch * epochs
    o, ep_ret, ep_len, total_test_ep_ret, num_episodes, is_succ = env.reset(), 0, 0, 0, 0, False
    logger.debug("train initial obs: {}".format(o))
    o, _, _, _ = preprocess(o, 0, False, None)
    agent = DDPGAgent(
        o, env.action_space, gamma, actor_lr, critic_lr,
        polyak, l2_action, logger
    )
    replay_buffer = ReplayBuffer(
        o.shape[0], env.action_space.shape[0], replay_size
    )

    for i in tqdm(range(int(total_steps))):
        if i > start_steps:
            a = agent.act(o, noise_scale)
        else:
            a = env.action_space.sample()

        o2, r, d, info = env.step(a)
        o2, r, d, info = preprocess(o2, r, d, info)
        replay_buffer.store(o, a, r, o2, d)
        ep_len += 1
        ep_ret += r
        is_succ |= bool(info["is_success"])

        # End of trajectory handling
        if d or (ep_len == max_ep_len):
            # logger.info(ep_len, ep_ret)
            num_episodes += 1
            writer.add_scalar("Train/return", ep_ret, num_episodes)
            writer.add_scalar("Train/steps", ep_len, num_episodes)
            writer.add_scalar("Train/is_succ", is_succ, num_episodes)
            o, ep_ret, ep_len, is_succ = env.reset(), 0, 0, False
            logger.debug("train reset initial obs: {}".format(o))
            o, _, _, _ = preprocess(o, 0, False, None)

        # Update handling
        if i >= update_after and i % update_every == 0:
            basis = i // update_every
            for j in range(update_every):
                batch = replay_buffer.sample_batch()
                loss_q, loss_pi, max_q = agent.update(batch)
                writer.add_scalar("Train/max_q", max_q, basis + j)
                writer.add_scalar("Train/loss_q", loss_q, basis + j)
                writer.add_scalar("Train/loss_pi", loss_pi, basis + j)

        # End of epoch handling
        if (i+1) % steps_per_epoch == 0:
            epoch = (i+1) // steps_per_epoch
            # logger.info(f"Epoch {epoch}\n-------------------------------")
            # logger.info(f"return: {ep_ret}   [{i:>7d}/{int(total_steps):>7d}]")
            test_ep_ret, test_ep_len, test_suc_rate = test(env_fn(), agent, preprocess, num_test_episodes, max_ep_len)
            writer.add_scalar("Test/return", test_ep_ret, epoch)
            writer.add_scalar("Test/steps", test_ep_len, epoch)
            writer.add_scalar("Test/succ_rate", test_suc_rate, epoch)
            total_test_ep_ret += test_ep_ret

    writer.close()
    return total_test_ep_ret

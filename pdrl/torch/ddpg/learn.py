import logging
import os
import torch
from statistics import mean
from gym.wrappers.record_video import RecordVideo
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from pdrl.torch.ddpg.agent import DDPGAgent
from pdrl.torch.ddpg.replay_memory import ReplayBuffer
from pdrl.torch.normalizer import Zscorer
from pdrl.utils.mpi import mpi_avg, num_procs, proc_id

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def test(test_env, agent, normalizer, pipeline, num_test_episodes, max_ep_len):
    ep_rets, ep_lens, is_successes = [], [], []

    for _ in range(num_test_episodes):
        f_o, d, rets, is_success, ep_len = test_env.reset(), False, [], False, 0
        logger.debug("test reset initial obs: {}".format(f_o))
        o, _, _, _, _, _ = pipeline.transform(f_o, None, 0, None, False, None)

        while(not d and (ep_len < max_ep_len) and not is_success):
            n_o = normalizer(o)
            a = agent.act(n_o, noise_scale=0, epsilon=0)
            f_o2, r, d, info = test_env.step(a)
            o, a, r, o2, d, info = pipeline.transform(f_o, a, r, f_o2, d, info)
            f_o, o = f_o2.copy(), o2.copy()
            rets.append(r)
            is_success |= bool(info["is_success"])
            ep_len += 1

        ep_rets.append(sum(rets))
        ep_lens.append(ep_len)
        is_successes.append(is_success)

    # logger.info(f"test return: {ep_ret}")
    test_env.reset()
    return mean(ep_rets), mean(ep_lens), mean(is_successes)


def learn(env_fn, pipeline, test_pipeline, epochs, steps_per_epoch, start_steps, update_after, update_every,
          num_test_episodes, max_ep_len, gamma, epsilon, actor_lr, critic_lr, replay_size, polyak, l2_action,
          noise_scale, batch_size, norm_clip, norm_eps, clip_return, is_pos_return, logdir=None, video=False):
    env = env_fn()
    test_env = env

    if proc_id() == 0:
        video_folder = os.path.join(logdir, "videos")
        test_env = RecordVideo(env, video_folder=video_folder) if video else env
        writer = SummaryWriter(logdir)

    total_steps = steps_per_epoch * epochs // num_procs()
    f_o, ep_ret, ep_len, total_test_ep_ret, num_episodes, is_succ = env.reset(), 0, 0, 0, 0, False
    logger.debug("train initial obs: {}".format(f_o))
    o, _, _, _, _, _ = pipeline.transform(f_o, None, 0, None, False, None)
    agent = DDPGAgent(
        o, env.action_space, gamma, actor_lr, critic_lr,
        polyak, l2_action, clip_return, is_pos_return, logger
    )
    normalizer = Zscorer(norm_clip, norm_eps)
    replay_buffer = ReplayBuffer(
        o.shape[1], env.action_space.shape[0], replay_size
    )

    # 入出力の型をモジュール毎に統一したい。
    for i in tqdm(range(int(total_steps))):

        if i > start_steps:
            n_o = normalizer(o)
            a = agent.act(n_o, noise_scale, epsilon)
        else:
            a = env.action_space.sample()

        f_o2, r, d, info = env.step(a)
        o, a, r, o2, d, info = pipeline.transform(f_o, a, r, f_o2, d, info)
        replay_buffer.store(o, a, r, o2, d)
        f_o = f_o2.copy()
        o = o2.copy()
        ep_len, ep_ret = ep_len + 1, ep_ret + r
        is_succ |= bool(info["is_success"])

        # End of trajectory handling
        if d or (ep_len == max_ep_len):
            # logger.info(ep_len, ep_ret)
            num_episodes += 1
            avg_ep_ret, avg_ep_len, avg_is_succ = mpi_avg(ep_ret), mpi_avg(ep_len), mpi_avg(is_succ)

            if proc_id() == 0:
                writer.add_scalar("Train/return", scalar_value=avg_ep_ret, global_step=num_episodes)
                writer.add_scalar("Train/steps", scalar_value=avg_ep_len, global_step=num_episodes)
                writer.add_scalar("Train/succ_rate", scalar_value=avg_is_succ, global_step=num_episodes)

            f_o, ep_ret, ep_len, is_succ = env.reset(), 0, 0, False
            logger.debug("train reset initial obs: {}".format(o))
            o, _, r, _,  d, _ = pipeline.transform(f_o, None, 0, None, False, None)

        # Update handling
        if i >= update_after and i % update_every == 0:
            basis = (i - update_after) // update_every
            for j in range(update_every):
                batch = replay_buffer.sample_batch(batch_size=batch_size)
                logger.debug("obs avg: {}, obs var: {}".format(torch.mean(batch["obs"], 0), torch.var(batch["obs"], 0)))
                logger.debug("obs2 avg: {}, obs2 var: {}".format(
                    torch.mean(batch["obs2"], 0), torch.var(batch["obs2"], 0)
                ))
                logger.debug("act avg: {}, act var: {}".format(torch.mean(batch["act"], 0), torch.var(batch["act"], 0)))
                batch["obs"] = normalizer(batch["obs"])
                batch["obs2"] = normalizer(batch["obs2"])
                logger.debug("obs avg: {}, obs var: {}".format(torch.mean(batch["obs"], 0), torch.var(batch["obs"], 0)))
                logger.debug("obs2 avg: {}, obs2 var: {}".format(
                    torch.mean(batch["obs2"], 0), torch.var(batch["obs2"], 0)
                ))
                loss_q, loss_pi, max_q = agent.update(batch)
                avg_loss_q, avg_loss_pi, avg_max_q = mpi_avg(loss_q), mpi_avg(loss_pi), mpi_avg(max_q)

                if proc_id() == 0:
                    writer.add_scalar("Train/q_avg", avg_max_q, basis + j)
                    writer.add_scalar("Train/loss_q", avg_loss_q, basis + j)
                    writer.add_scalar("Train/loss_pi", avg_loss_pi, basis + j)

            agent.sync_target()

        # End of epoch handling
        if (i+1) % steps_per_epoch == 0:
            epoch = (i+1) // steps_per_epoch
            # logger.info(f"Epoch {epoch}\n-------------------------------")
            # logger.info(f"return: {ep_ret}   [{i:>7d}/{int(total_steps):>7d}]")
            test_ep_ret, test_ep_len, test_suc_rate = test(
                test_env, agent, normalizer, test_pipeline, num_test_episodes, max_ep_len
            )
            # TODO Actually, suc_rate should be calculated by harmonic mean.
            avg_test_ep_ret = mpi_avg(test_ep_ret)
            avg_test_ep_len = mpi_avg(test_ep_len)
            avg_test_suc_rate = mpi_avg(test_suc_rate)

            if proc_id() == 0:
                writer.add_scalar("Test/return", avg_test_ep_ret, epoch)
                writer.add_scalar("Test/steps", avg_test_ep_len, epoch)
                writer.add_scalar("Test/succ_rate", avg_test_suc_rate, epoch)
            total_test_ep_ret += test_ep_ret

    if proc_id() == 0:
        writer.close()

    return total_test_ep_ret

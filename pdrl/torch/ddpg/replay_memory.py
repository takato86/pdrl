import logging
import numpy as np
import torch
from pdrl.utils.constants import device


logger = logging.getLogger()


def create_replay_buffer_fn(shaper, size):

    def create_replay_buffer(obs_dim, act_dim):
        """return a replay buffer instance.

        Args:
            obs_dim (_type_): _description_
            act_dim (_type_): _description_
            size (_type_): _description_
            shaper (_type_, optional): _description_. Defaults to None.

        Returns:
            _type_: _description_
        """
        if shaper is None:
            # Case without shaping
            logger.info("Replay Buffer is selected.")
            replay_buffer = ReplayBuffer(obs_dim, act_dim, size)
        elif shaper.is_learn:
            # Case with shaping
            logger.info("Dynamic Shaping Replay Buffer is selected.")
            replay_buffer = DynamicShapingReplayBuffer(obs_dim, act_dim, size, shaper)
        else:
            logger.info("Static Shaping Replay Buffer is selected.")
            replay_buffer = StaticShapingReplayBuffer(obs_dim, act_dim, size, shaper)

        return replay_buffer

    return create_replay_buffer


def combined_shape(length, shape=None):
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)


class ReplayBuffer:
    """
    A simple FIFO experience replay buffer for DDPG agents.
    """

    def __init__(self, obs_dim, act_dim, size):
        self.obs_buf = np.zeros(combined_shape(size, obs_dim), dtype=np.float32)
        self.obs2_buf = np.zeros(combined_shape(size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros(combined_shape(size, act_dim), dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.ptr, self.size, self.max_size = 0, 0, size

    def store(self, obs, act, rew, next_obs, done, info):
        self.obs_buf[self.ptr] = obs
        self.obs2_buf[self.ptr] = next_obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr+1) % self.max_size
        self.size = min(self.size+1, self.max_size)

    def sample_batch(self, batch_size=32):
        idxs = np.random.randint(0, self.size, size=batch_size)
        batch = dict(obs=self.obs_buf[idxs],
                     obs2=self.obs2_buf[idxs],
                     act=self.act_buf[idxs],
                     rew=self.rew_buf[idxs],
                     done=self.done_buf[idxs])
        return {
            k: torch.as_tensor(v, dtype=torch.float32, device=device)
            for k, v in batch.items()
        }


class StaticShapingReplayBuffer(ReplayBuffer):
    def __init__(self, obs_dim, act_dim, size, shaper):
        super().__init__(obs_dim, act_dim, size)
        self.shaper = shaper

    def sample_batch(self, batch_size=32):
        idxs = np.random.randint(0, self.size, size=batch_size)
        shaping = np.zeros(batch_size)

        # shapingの報酬値計算
        for i, (aobs, aobs2) in enumerate(zip(self.obs_buf[idxs], self.obs2_buf[idxs])):
            shaping[i] = self.shaper.shape(aobs, aobs2)

        batch = dict(obs=self.obs_buf[idxs],
                     obs2=self.obs2_buf[idxs],
                     act=self.act_buf[idxs],
                     rew=self.rew_buf[idxs] + shaping,
                     done=self.done_buf[idxs])
        return {
            k: torch.as_tensor(v, dtype=torch.float32, device=device)
            for k, v in batch.items()
        }


class DynamicShapingReplayBuffer:
    """
    A simple FIFO experience replay buffer for DDPG agents with SRS with dynamic potential.
    """

    def __init__(self, obs_dim, act_dim, size, shaper):
        self.obs_buf = np.zeros(combined_shape(size, obs_dim), dtype=np.float32)
        self.obs2_buf = np.zeros(combined_shape(size, obs_dim), dtype=np.float32)
        self.aobs_buf = np.zeros(size, dtype=np.float32)
        self.aobs2_buf = np.zeros(size, dtype=np.float32)
        self.act_buf = np.zeros(combined_shape(size, act_dim), dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.ptr, self.size, self.max_size, self.shaper = 0, 0, size, shaper

    def store(self, obs, act, rew, next_obs, done, info):
        self.obs_buf[self.ptr] = obs
        self.obs2_buf[self.ptr] = next_obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr+1) % self.max_size
        self.size = min(self.size+1, self.max_size)
        self.aobs_buf[self.ptr] = self.shaper.current_state
        _ = self.shaper.step(obs, act, rew, next_obs, done, info)
        # inner state of shaper transits when the `shape` method is called.
        if not done:
            self.aobs2_buf[self.ptr] = self.shaper.current_state
        else:
            self.aobs2_buf[self.ptr] = self.aobs_buf[self.ptr]

    def sample_batch(self, batch_size=32):
        idxs = np.random.randint(0, self.size, size=batch_size)
        shaping = np.zeros(batch_size)

        # shapingの報酬値計算
        for i, (aobs, aobs2) in enumerate(zip(self.aobs_buf[idxs], self.aobs2_buf[idxs])):
            shaping[i] = self.shaper.shape(aobs, aobs2)

        batch = dict(obs=self.obs_buf[idxs],
                     obs2=self.obs2_buf[idxs],
                     act=self.act_buf[idxs],
                     rew=self.rew_buf[idxs] + shaping,
                     done=self.done_buf[idxs])
        return {
            k: torch.as_tensor(v, dtype=torch.float32, device=device)
            for k, v in batch.items()
        }

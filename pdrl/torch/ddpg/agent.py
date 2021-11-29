"""DDPGの実装"""
import torch
from torch.optim import Adam
import copy
import numpy as np
from pdrl.torch.agent import Agent
from pdrl.torch.ddpg.network import ActorCritic


class DDPGAgent(Agent):
    def __init__(self, observation_space, action_space, gamma, actor_lr, critic_lr, polyak, logger):
        self.gamma = gamma
        self.actor_critic = ActorCritic(observation_space, action_space)
        self.target_ac = copy.deepcopy(self.actor_critic)
        self.actor_optimizer = Adam(self.actor_critic.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = Adam(self.actor_critic.critic.parameters(), lr=critic_lr)
        self.polyak = polyak
        self.act_dim = action_space.shape[0]
        self.logger = logger

    def act(self, observation, noise_scale):
        action = self.actor_critic.act(
            torch.as_tensor(observation, dtype=torch.float32)
        )
        action += noise_scale * np.random.random(self.act_dim)
        return action

    def compute_Q_loss(self, datum):
        o, a, r, o2, d = datum['obs'], datum['act'], datum['rew'], datum['obs2'], datum['done']
        q_value = self.actor_critic.critic(o, a)

        with torch.no_grad():
            q_pi_target = self.target_ac.critic(o2, self.target_ac.actor(o2))
            backup = r + self.gamma * (1 - d) * q_pi_target

        loss_q = ((q_value - backup)**2).mean()
        # "detach" Returns a new Tensor, detached from the current graph.
        loss_info = dict(QVals=q_value.detach().numpy())
        return loss_q, loss_info

    def compute_pi_loss(self, datum):
        o = datum['obs']
        q_pi = self.actor_critic.critic(o, self.actor_critic.actor(o))
        return -q_pi.mean()

    def update(self, datum):
        # Critic Networkの更新処理
        self.critic_optimizer.zero_grad()
        loss_q, loss_info = self.compute_Q_loss(datum)
        loss_q.backward()
        self.critic_optimizer.step()

        # Freeze Critic Network
        for p in self.actor_critic.critic.parameters():
            p.requires_grad = False

        self.actor_optimizer.zero_grad()
        loss_pi = self.compute_pi_loss(datum)
        loss_pi.backward()
        self.actor_optimizer.step()

        for p in self.actor_critic.critic.parameters():
            p.requires_grad = True

        with torch.no_grad():
            for p, p_target in zip(self.actor_critic.parameters(), self.target_ac.parameters()):
                p_target.data.mul_(self.polyak)
                p_target.data.add_((1 - self.polyak) * p.data)

"""DDPGの実装"""
import torch
from torch.optim import Adam
import copy
import numpy as np
from pdrl.torch.agent import Agent
from pdrl.torch.ddpg.network import ActorCritic
from pdrl.utils.mpi_torch import mpi_avg_grad, sync_params



class DDPGAgent(Agent):
    def __init__(self, observation_space, action_space, gamma, actor_lr, critic_lr, polyak, l2_action, logger):
        self.gamma = gamma
        self.actor_critic = ActorCritic(observation_space, action_space)
        # MPIプロセス間で重みを共通化
        sync_params(self.actor_critic.actor)
        sync_params(self.actor_critic.critic)
        self.target_ac = copy.deepcopy(self.actor_critic)
        self.actor_optimizer = Adam(self.actor_critic.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = Adam(self.actor_critic.critic.parameters(), lr=critic_lr)
        self.polyak = polyak
        self.act_dim = action_space.shape[0]
        self.max_act = torch.Tensor(action_space.high)
        self.logger = logger
        self.l2_action = l2_action

    def act(self, observation, noise_scale):
        # TODO obsの正規化
        action = self.actor_critic.act(
            torch.as_tensor(observation, dtype=torch.float32)
        )
        action += noise_scale * self.max_act.numpy() * np.random.random(self.act_dim)
        # TODO epsilon-greedyの実装
        return action

    def compute_Q_loss(self, datum):
        o, a, r, o2, d = datum['obs'], datum['act'], datum['rew'], datum['obs2'], datum['done']
        q_value = self.actor_critic.critic(o, a / self.max_act)

        with torch.no_grad():
            # ターゲットネットワークの更新をしないようにする。
            q_pi_target = self.target_ac.critic(o2, self.target_ac.actor(o2) / self.max_act)
            backup = r + self.gamma * (1 - d) * q_pi_target

        loss_q = ((q_value - backup)**2).mean()
        # "detach" Returns a new Tensor, detached from the current graph.
        loss_info = dict(QVals=q_value.detach().numpy())
        return loss_q, loss_info

    def compute_pi_loss(self, datum):
        o = datum['obs']
        normalized_a_pi = self.actor_critic.actor(o) / self.max_act
        q_pi = self.actor_critic.critic(o, normalized_a_pi)
        loss_pi = -q_pi.mean()
        # 正則化項
        loss_pi += self.l2_action * (normalized_a_pi**2).mean()
        return loss_pi

    def update(self, datum):
        """方策と価値関数の更新"""
        # Critic Networkの更新処理
        # TODO datum["obs"]の正規化
        self.critic_optimizer.zero_grad()
        loss_q, loss_q_info = self.compute_Q_loss(datum)
        qs = loss_q_info["QVals"]
        loss_q.backward()
        # ここで平均化
        mpi_avg_grad(self.actor_critic.critic)
        self.critic_optimizer.step()

        # Freeze Critic Network
        for p in self.actor_critic.critic.parameters():
            p.requires_grad = False

        self.actor_optimizer.zero_grad()
        loss_pi = self.compute_pi_loss(datum)
        loss_pi.backward()
        # ここで勾配を平均化
        mpi_avg_grad(self.actor_critic.actor)
        self.actor_optimizer.step()

        for p in self.actor_critic.critic.parameters():
            p.requires_grad = True

        loss_q_numpy, loss_pi_numpy = loss_q.detach().numpy(), loss_pi.detach().numpy()
        return loss_q_numpy, loss_pi_numpy, max(qs)


    def sync_target(self):
        """Target Networkとの同期"""
        with torch.no_grad():
            for p, p_target in zip(self.actor_critic.parameters(), self.target_ac.parameters()):
                p_target.data.mul_(self.polyak)
                p_target.data.add_((1 - self.polyak) * p.data)
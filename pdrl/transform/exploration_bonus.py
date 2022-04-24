import torch
from torch import nn
from torch.optim import Adam
from pdrl.transform.pipeline import Step
from pdrl.utils.mpi_torch import mpi_avg_grad


def create_bonus(configs, env_fn):
    pass


class ExplorationBonusStep(Step):
    """Exploration bonus for usage in pipepline

    """

    def transform(self, pre_obs, pre_action, r, obs, d, info):
        pass


class TargetRandomNetwork(nn.Module):
    def __init__(self, d_input, d_output):
        self.network = nn.Sequential(
            nn.Linear(d_input, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, d_output)
        )

    def forward(self, obs):
        return self.network(obs)


class Predictor(nn.Module):
    def __init__(self, d_input, d_output):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(d_input, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, d_output)
        )

    def forward(self, obs):
        return self.network(obs)


class RND:
    """Random Network Distillation.
    """
    def __init__(self, observation_space, feature_size, lr):
        input_dim = observation_space.shape[1]
        self.target = TargetRandomNetwork(input_dim, feature_size)
        self.predictor = Predictor(input_dim, feature_size)
        self.predictor_optimizer = Adam(self.predictor.parameters(), lr=lr)

    def update(self, datum):
        self.predictor_optimizer.zero_grad()
        target_feats = self.target(datum)
        predictor_feats = self.predictor(datum)
        mse_loss = torch.mean((target_feats - predictor_feats)**2)
        mse_loss.backward()
        mpi_avg_grad(self.predictor)
        self.predictor_optimizer.step()

    def act(self, obs):
        target_feature = self.target(obs)
        predictor_feature = self.predictor(obs)
        intrinsic_reward = torch.mean((target_feature - predictor_feature)**2)
        return intrinsic_reward

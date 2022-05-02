from pdrl.torch.td3.train import train as train_td3
from pdrl.torch.td3.optimize import optimize_hyparams as optimize_td3
from pdrl.torch.ddpg.train import train as train_ddpg
from pdrl.torch.ddpg.optimize import optimize_hyparams as optimize_ddpg


TRAIN_FNS = {
    "td3": train_td3,
    "ddpg": train_ddpg
}


OPTIMIZE_FNS = {
    "td3": optimize_td3,
    "ddpg": optimize_ddpg,
}

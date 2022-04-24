

def sample_shaping_params(trial, method):
    hyperparams = {}
    if method is not None:
        hyperparams = HYPERPARAMETER_SAMPLER[method](trial)
    return hyperparams


def sample_dta_params(trial):
    hyperparams = {
        "gamma": trial.user_attrs["GAMMA"],
        "lr": trial.suggest_categorical("critic_lr_dta", [0.001, 0.0001, 0.005, 0.01, 0.00001]),
        "aggr_id": "dta",
        "vid": "table",
        "values": {
            0: 0.1,
            1: 0.3,
            2: 0.5
        }
    }
    return hyperparams


HYPERPARAMETER_SAMPLER = {
    "dta": sample_dta_params
}

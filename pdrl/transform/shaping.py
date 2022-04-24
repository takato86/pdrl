import shaner
from pdrl.transform.pipeline import Step
from pdrl.experiments.pick_and_place.subgoal import generate_subgoals
from pdrl.experiments.pick_and_place.is_success import is_success
from pdrl.experiments.pick_and_place.achiever import FetchPickAndPlaceAchiever


SHAPING_ALGS = {
    "dta": shaner.SarsaRS
}


def create_shaper(configs, env_fn):
    shaping_method = configs.get("shaping_method")
    if shaping_method is not None:
        subgoals = generate_subgoals()
        achiever_params = configs["achiever_params"]
        # TODO implementation by domain agnostic way.
        achiever = FetchPickAndPlaceAchiever(
            subgoals=subgoals,
            **achiever_params
        )
        shaper = SHAPING_ALGS[shaping_method](
            abstractor=achiever,
            is_success=is_success,
            **configs["shaping_params"]
        )
        return shaper
    else:
        return None


class ShapingStep(Step):
    """shaping class for usage in pipeline

    Args:
        Step (_type_): _description_
    """
    def __init__(self, shaper):
        self.shaper = shaper

    def transform(self, pre_obs, pre_action, r, obs, d, info):
        is_none = [
            x is None for x in [pre_obs, pre_action, r, obs, d]
        ]
        if not any(is_none):
            # Noneが一つでもある場合はスキップ。
            f = self.shaper.step(pre_obs, pre_action, r, obs, d, info)
            r += f

        return pre_obs, pre_action, r, obs, d, info

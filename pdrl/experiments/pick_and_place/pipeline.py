import numpy as np
import shaner
from pdrl.transform.shaping import ShapingStep
from pdrl.transform.pipeline import Pipeline
from pdrl.experiments.pick_and_place.preprocess import RoboticsObservationTransformer
from pdrl.experiments.pick_and_place.achiever import FetchPickAndPlaceAchiever
from pdrl.experiments.pick_and_place.is_success import is_success


SHAPING_ALGS = {
    "dta": shaner.SarsaRS
}


def create_test_pipeline(configs):
    return Pipeline([RoboticsObservationTransformer])


def create_pipeline(configs):
    shaping_method = configs.get("shaping_method")

    if shaping_method is not None:
        subgoals = generate_subgoals()
        achiever_params = configs["achiever_params"]
        achiever = FetchPickAndPlaceAchiever(
            subgoals=subgoals,
            **achiever_params
        )
        shaper = SHAPING_ALGS[shaping_method](
            abstractor=achiever,
            is_success=is_success,
            **configs["shaping_params"]
        )
        shaping = ShapingStep(shaper)
        pipe = Pipeline([RoboticsObservationTransformer, shaping])
    else:
        pipe = Pipeline([RoboticsObservationTransformer])

    return pipe


def generate_subgoals():
    # Subgoal1: Objectの絶対座標[x,y,z] = achieved_goal
    # Subgoal2: Objectの絶対座標とArmの位置が同じでアームを閉じている状態。
    subgoal1 = np.full(28, np.nan)
    # subgoal1[6:8] = [0, 0]
    subgoal1[6:9] = [0, 0, 0]
    subgoal2 = np.full(28, np.nan)
    # subgoal2[6:9] = [0, 0, 0]
    subgoal2[6:11] = [0, 0, 0, 0.02, 0.02]
    return [subgoal1, subgoal2]

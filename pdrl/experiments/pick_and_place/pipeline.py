from pdrl.experiments.pick_and_place.achiever import FetchPickAndPlaceAchieverStep
from pdrl.experiments.pick_and_place.subgoal import generate_subgoals
from pdrl.transform.pipeline import Pipeline
from pdrl.experiments.pick_and_place.preprocess import RoboticsObservationTransformer


def create_test_pipeline(configs):
    subgoals = generate_subgoals()
    return Pipeline(
        [
            RoboticsObservationTransformer(),
            FetchPickAndPlaceAchieverStep(configs["achiever_params"]["_range"], subgoals)
        ]
    )


def create_pipeline(configs):
    subgoals = generate_subgoals()
    pipe = Pipeline(
        [
            RoboticsObservationTransformer(),
            FetchPickAndPlaceAchieverStep(configs["achiever_params"]["_range"], subgoals)
        ]
    )
    # shaping_method = configs.get("shaping_method")
    # Move the shaping step into Experience Replay class.
    #
    # if shaping_method is not None:
    #     subgoals = generate_subgoals()
    #     achiever_params = configs["achiever_params"]
    #     achiever = FetchPickAndPlaceAchiever(
    #         subgoals=subgoals,
    #         **achiever_params
    #     )
    #     shaper = SHAPING_ALGS[shaping_method](
    #         abstractor=achiever,
    #         is_success=is_success,
    #         **configs["shaping_params"]
    #     )
    #     shaping = ShapingStep(shaper)
    #     pipe = Pipeline([RoboticsObservationTransformer, shaping])
    # else:
    #     pipe = Pipeline([RoboticsObservationTransformer])

    return pipe

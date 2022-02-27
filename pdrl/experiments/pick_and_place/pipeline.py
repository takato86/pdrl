from pdrl.transform.pipeline import Pipeline
from pdrl.experiments.pick_and_place.preprocess import RoboticsObservationTransformer


def create_test_pipeline(configs):
    return Pipeline([RoboticsObservationTransformer])


def create_pipeline(configs):
    pipe = Pipeline([RoboticsObservationTransformer])
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

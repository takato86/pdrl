from pdrl.transform.pipeline import Step


class ShapingStep(Step):
    def __init__(self, shaper):
        self.shaper = shaper

    def transform(self, pre_obs, pre_action, r, obs, d, info):
        is_none = [
            x is None for x in [pre_obs, pre_action, r, obs, d]
        ]
        if not any(is_none):
            # Noneが一つでもある場合はスキップ。
            f = self.shaper.shape(pre_obs, pre_action, r, obs, d, info)
            r += f

        return pre_obs, pre_action, r, obs, d, info

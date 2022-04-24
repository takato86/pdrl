import numpy as np


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

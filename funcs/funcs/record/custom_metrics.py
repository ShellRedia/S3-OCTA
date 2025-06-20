from skimage.morphology import skeletonize
import numpy as np


def clDice(v_p, v_l):
    """[this function computes the cldice metric]

    Args:
        v_p ([bool]): [predicted image]
        v_l ([bool]): [ground truth image]

    Returns:
        [float]: [cldice metric]
    """
    epsilon = 1e-8
    cl_score = lambda v, s: np.sum(v*s)/np.sum(s)
    
    tprec = cl_score(v_p,skeletonize(v_l))
    tsens = cl_score(v_l,skeletonize(v_p))

    return 2*tprec*tsens/(tprec+tsens+epsilon)
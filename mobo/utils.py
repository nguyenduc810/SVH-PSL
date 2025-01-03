from time import time
import numpy as np
# from pymoo.factory import get_performance_indicator
from pymoo.indicators.hv import HV
import random
import torch


class Timer:
    '''
    For time recording and message logging
    '''
    def __init__(self):
        self.t = time()

    def log(self, string=None, reset=True):
        msg = '%.2fs' % (time() - self.t)

        if string is not None:
            msg = string + ': ' + msg
        print(msg)
        
        if reset:
            self.t = time()

    def reset(self):
        self.t = time()

def set_seed(seed):
    """for reproducibility
    :param seed:
    :return:
    """
    np.random.seed(seed)
    random.seed(seed)

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def find_pareto_front(Y, return_index=False):
    '''
    Find pareto front (undominated part) of the input performance data.
    '''
    if len(Y) == 0: return np.array([])
    sorted_indices = np.argsort(Y.T[0])
    pareto_indices = []
    for idx in sorted_indices:
        # check domination relationship
        if not (np.logical_and((Y <= Y[idx]).all(axis=1), (Y < Y[idx]).any(axis=1))).any():
            pareto_indices.append(idx)
    pareto_front = Y[pareto_indices].copy()

    if return_index:
        return pareto_front, pareto_indices
    else:
        return pareto_front


def calc_hypervolume(pfront, ref_point):
    '''
    Calculate hypervolume of pfront based on ref_point
    '''
    hv = HV(ref_point=ref_point)
    return hv(pfront)


def safe_divide(x1, x2):
    '''
    Divide x1 / x2, return 0 where x2 == 0
    '''
    return np.divide(x1, x2, out=np.zeros(np.broadcast(x1, x2).shape), where=(x2 != 0))


def expand(x, axis=-1):
    '''
    Concise way of expand_dims
    '''
    return np.expand_dims(x, axis=axis)
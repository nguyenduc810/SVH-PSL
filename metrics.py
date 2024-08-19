import numpy as np
import torch


def mean_Euclidean_dist(y_hat, y_truth):
    return np.mean(np.linalg.norm(y_truth - y_hat, axis = 1))

def normalize(x):
    return [float(i)/max(x) for i in x]
#!/usr/bin/env python
# Partitioning the population by Local PCA algorithm

# This function is translated from the Matlab code in
# http://bimk.ahu.edu.cn/index.php?s=/Index/Software/index.html


from DA_module.LocalPCA import *


def RMMEDA_operator(PopDec, K, M, XLow, XUpp):
    N, D = PopDec.shape
    ## Modeling
    Model, probability = LocalPCA(PopDec, M, K)
    ## Reproduction
    OffspringDec = np.zeros((5000, D))
    # Generate new trial solutions one by one
    for i in np.arange(5000):
        # Select one cluster by Roulette-wheel selection
        k = (np.where(np.random.rand() <= probability))[0][0]
        # Generate one offspring
        if not len(Model[k]['eVector']) == 0:
            lower = Model[k]['a'] - 0.25 * (Model[k]['b'] - Model[k]['a'])
            upper = Model[k]['b'] + 0.25 * (Model[k]['b'] - Model[k]['a'])
            trial = np.random.uniform(0, 1) * (upper - lower) + lower  # ,(1,M-1)
            trial = trial.T
            sigma = np.sum(np.abs(Model[k]['eValue'][M - 1:D])) / (D - M + 1)
            a = Model[k]['eVector'][:, :M - 1].conj().transpose()
            c = np.dot(trial,a)
            OffspringDec[i, :] = Model[k]['mean'] + c + np.random.randn(D) * np.sqrt(sigma)
        else:
            OffspringDec[i, :] = Model[k]['mean'] + np.random.randn(D)
        NN, D = OffspringDec.shape
        low = np.tile(XLow, (NN, 1))
        upp = np.tile(XUpp, (NN, 1))
        lbnd = OffspringDec <= low
        ubnd = OffspringDec >= upp
        OffspringDec[lbnd] = low[lbnd]
        OffspringDec[ubnd] = upp[ubnd]
        # OffspringDec[lbnd] = 0.5 * (PopDec[lbnd] + low[lbnd])
        # OffspringDec[ubnd] = 0.5 * (PopDec[ubnd] + upp[ubnd])
    return OffspringDec
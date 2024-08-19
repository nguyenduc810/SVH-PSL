
import numpy as np
import torch
import pickle
from DA_module.problem import get

from lhs import lhs
from pymoo.indicators.hv import HV
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from mobo.surrogate_model.gaussian_process import GaussianProcess
from mobo.transformation import StandardTransform

from DA_module.spea2_env import environment_selection
from DA_module.PM_mutation import pm_mutation
from DA_module.GAN_model import GAN
from DA_module.Generate import RMMEDA_operator
# from pymop.factory import get_problem
from pymoo.problems import get_problem
from DA_module.mating_selection import random_genetic_variation
from DA_module.evolution.utils import *
from DA_module.learning.model_init import *
from DA_module.learning.model_update import *
from DA_module.learning.prediction import *
import os
from mobo.utils import set_seed
set_seed(45)
# -----------------------------------------------------------------------------
mmm = []
ins_list = [
              'zdt3'
            ]

# number of independent runs
n_run = 5
# number of initialized solutions
n_init = 20
# number of iterations, and batch size per iteration
n_iter = 20
n_sample = 5

# coefficient of LCB
coef_lcb = 0.1

# number of sampled candidates on the approxiamte Pareto front
n_candidate = 1000 
# number of optional local search

def get_device(no_cuda=False, gpus=None):
    return torch.device(f"cuda:{gpus}" if torch.cuda.is_available() and not no_cuda else "cpu")
device=get_device(gpus ='0')

# -----------------------------------------------------------------------------

hv_list = {}
import pandas as pd

dic = {'zdt1':[0.9994, 6.0576],
        'zdt2':[0.9994, 6.8960],
        'zdt3':[0.9994, 6.0571],
        'dtlz2':[2.8390, 2.9011, 2.8575],
        'RE21':[2886.3695604236013, 0.039999999999998245],
       'RE32':[ 37.7831517014 , 17561.6 ,425062976.628],
       'RE33':[5.3067 , 3.12833430979 , 25.0 ],
       'RE36':[5.931, 56.0, 0.355720675227],
       'RE37':[0.98949120096, 0.956587924661 , 0.987530948586],
       'RE41':[39.2905121788, 4.42725, 13.09138125 ,9.49401929991],
       'VLMOP2': [1.1,1.1],
       }

for test_ins in ins_list:
    print(test_ins)
    # get problem info
    hv_all_value = np.zeros([n_run, n_iter+1])
    if test_ins.startswith('zdt'):
        problem = get_problem(test_ins, n_var=20)
    else:
        problem = get(test_ins)

    n_dim = problem.n_var
    n_obj = problem.n_obj
    lbound = torch.zeros(n_dim).float()
    ubound = torch.ones(n_dim).float()
    ref_point = dic[test_ins]

    if not os.path.exists(f"result/DA/logs_{coef_lcb}_{test_ins}_{n_obj}_{n_dim}_nsamples{n_sample}"):
        os.makedirs(f"result/DA/logs_{coef_lcb}_{test_ins}_{n_obj}_{n_dim}_nsamples{n_sample}")

    # repeatedly run the algorithm n_run times
    for run_iter in range(n_run):
        x_init = lhs(n_dim, n_init)
        y_init = problem.evaluate(x_init)
        # print(y_init)
        p_rel_map, s_rel_map = init_dom_rel_map(300)
        p_model = init_dom_nn_classifier(x_init, y_init, p_rel_map, pareto_dominance, problem, device=device)  # init Pareto-Net
        # print(p_model)
        evaluated = len(y_init)

        X = x_init
        Y = y_init

        net = GAN(n_dim, 30, 0.0001, 200, n_dim)
        z = torch.zeros(n_obj).to(device)
        i_iter = 0
        while True:
            transformation = StandardTransform([0, 1])
            # print(Y)
            transformation.fit(X, Y)
            X_norm, Y_norm = transformation.do(X, Y)
            _, index = environment_selection(Y, len(X)//3)
            real = X[index, :]
            label = np.zeros((len(Y), 1))
            label[index, :] = 1
            net.train(X, label, real)
            surrogate_model = GaussianProcess(n_dim, n_obj, nu=5)
            surrogate_model.fit(X_norm, Y_norm)

            nds = NonDominatedSorting()
            idx_nds = nds.do(Y_norm)

            Y_nds = Y_norm[idx_nds[0]]

            X_gan = net.generate(real / np.tile(ubound, (np.shape(real)[0], 1)), n_init*10) * \
                          np.tile(ubound, (n_init*10, 1))
            X_gan = pm_mutation(X_gan, [lbound, ubound])
            X_ga = random_genetic_variation(real, 1000,list(np.zeros(n_dim)),list(np.ones(n_dim)),n_dim)
            X_gan = np.concatenate((X_ga, X_gan), axis=0)

            p_dom_labels, p_cfs = nn_predict_dom_inter(X_gan, real, p_model, device)

            
            res = np.sum(p_dom_labels,axis=1)
            iindex = np.argpartition(-res, 100)
            result_args = iindex[:100]
            X_dp = X_gan[result_args,:]

            X_psl = RMMEDA_operator(np.concatenate((X_dp, real), axis=0), 5, n_obj, lbound, ubound)


            Y_candidate_mean = surrogate_model.evaluate(X_psl)['F']


            Y_candidata_std = surrogate_model.evaluate(X_psl, std=True)['S']

            
            Y_candidate = Y_candidate_mean - coef_lcb * Y_candidata_std
            Y_candidate_mean = Y_candidate
            
            
            best_subset_list = []
            Y_p = Y_nds
            for b in range(n_sample):
                hv = HV(ref_point=np.max(np.vstack([Y_p, Y_candidate_mean]), axis=0))
                best_hv_value = 0
                best_subset = None

                for k in range(len(Y_candidate_mean)):
                    Y_subset = Y_candidate_mean[k]
                    Y_comb = np.vstack([Y_p, Y_subset])
                    hv_value_subset = hv(Y_comb)
                    if hv_value_subset > best_hv_value:
                        best_hv_value = hv_value_subset
                        best_subset = [k]

                Y_p = np.vstack([Y_p, Y_candidate_mean[best_subset]])
                best_subset_list.append(best_subset)
            best_subset_list = np.array(best_subset_list).T[0]

            X_candidate = X_psl
            X_new = X_candidate[best_subset_list]

            Y_new = problem.evaluate(X_new)
            Y_new = torch.tensor(Y_new).to(device)

            X_new = torch.tensor(X_new).to(device)
            X = np.vstack([X, X_new.detach().cpu().numpy()])
            Y = np.vstack([Y, Y_new.detach().cpu().numpy()])

            update_dom_nn_classifier(p_model, X, Y, p_rel_map, pareto_dominance, problem, device=device)



            hv = HV(ref_point=np.array(ref_point))
            hv_value = hv(Y)
            hv_all_value[run_iter, i_iter] = hv_value
            i_iter = i_iter+1
            print("hv", "{:.4e}".format(np.mean(hv_value)))
            print("***")
            evaluated = evaluated + n_sample
            if i_iter == 20:
                break
        np.save(f"result/DA/logs_{coef_lcb}_{test_ins}_{n_obj}_{n_dim}_nsamples{n_sample}/evaluation_{test_ins}_X_{n_dim}_{run_iter}", X)
        np.save(f"result/DA/logs_{coef_lcb}_{test_ins}_{n_obj}_{n_dim}_nsamples{n_sample}/evaluation_{test_ins}_Y_{n_dim}_{run_iter}", Y)

        hv_list[test_ins] = hv_all_value
        print("************************************************************")







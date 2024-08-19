"""
Runing the proposed Paret Set Learning (PSL) method on 15 test problems.
"""
import argparse
import numpy as np
import torch
import pickle
import cloudpickle
import os
import time
import pdb
from problem import get_problem
from partitioning import sampling_vector_randomly, sampling_vector_evenly
from pymoo.problems import get_problem as get_problem_pymoo

from lhs import lhs

from pymoo.indicators.hv import HV
# from pymoo.factory import get_performance_indicator as HV

from pymoo.config import Config
Config.warnings ['not_compiled'] = False

from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting

from mobo.surrogate_model.gaussian_process import GaussianProcess
from mobo.transformation import StandardTransform

from model import ParetoSetModel

import random

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



# -----------------------------------------------------------------------------


ins_list = ['zdt3']
# , 'F2', 'RE33', 'RE36', 'RE37']
# ins_list = ['RE36']

# number of initialized solutions
n_init = 20
# number of iterations, and batch size per iteration
n_iter = 20
n_run = 10

n_sample = 5
 
# PSL 
# number of learning steps
n_steps = 1000 
lr = 1e-3
# coefficient of LCB
coef_lcb = 0.1

# number of preference region and preference vector per region
n_region = 20
n_candidate_per_region = 5

# number of sampled candidates on the aproxiamte Pareto front
# n_candidate = n_region * n_candidate_per_region  
n_candidate = 1000 
# number of optional local search
n_local = 0

# device
device = 'cuda:0' # 'cuda:1'

dic = {'zdt1':[0.9994, 6.0576],
        'zdt2':[0.9994, 6.8960],
        'zdt3':[0.9994, 6.0571],
        'dtlz2':[2.8390, 2.9011, 2.8575],
        'RE21':[2886.3695604236013, 0.039999999999998245],
       'RE32':[ 37.7831517014 , 17561.6 ,425062976.628],
       'RE33':[5.3067 , 3.12833430979 , 25.0 ],
       'RE36':[5.931, 56.0, 0.355720675227],
       'RE37':[0.98949120096, 0.956587924661 , 0.987530948586],
       'RE41':[39.2905121788, 4.42725 ,13.09138125 ,9.49401929991],
       'VLMOP2': [1.1,1.1],
       }
# -----------------------------------------------------------------------------

hv_list = {}
for test_ins in ins_list:
    # print(test_ins)
    set_seed(42)
    
    if test_ins in ['VLMOP2']:
        n_dim = 6
    elif test_ins in ['zdt1','zdt2','zdt3']:
        n_dim = 20

    start = time.time()
    
        
    # get problem info
    hv_all_value = np.zeros([n_run, n_iter])
    if test_ins.startswith('zdt'):
        problem = get_problem_pymoo(test_ins, n_var=n_dim)
        n_dim = problem.n_var
        n_obj = problem.n_obj
    else:
        problem = get_problem(test_ins)
        n_dim = problem.n_dim
        n_obj = problem.n_obj

    suffix=f"PSL-MOBO_n_samples_{n_sample}"
    suffix_dir = ""
    
    if not os.path.exists(f"result/PSL_MOBO/batch_{n_sample}/logs_{coef_lcb}_{test_ins}_{n_obj}_{n_dim}_{suffix}"):
        os.makedirs(f"result/PSL_MOBO/batch_{n_sample}/logs_{coef_lcb}_{test_ins}_{n_obj}_{n_dim}_{suffix}")
    
    n_region_vec = n_obj
    
    n_pref_update = 10
    
    # number of reference vector for testing and list of preference vectors
    n_test = 100 * n_obj
    
    pref_vec_test = sampling_vector_evenly(n_obj, n_test)
    
    print(f"Problem: {test_ins}\nN dim: {n_dim} \nN objective: {n_obj} \nLogs dir: logs_{test_ins}{suffix}/\nRun: {suffix}")
    

    print("***********************************************")

    ref_point = dic[test_ins]

    
    for run_iter in range(n_run):
        front_list, x_list, y_list = {}, {}, {}

        front_list = np.zeros((n_iter, n_test, n_obj))
        x_list = np.zeros((n_iter, n_test, n_dim))
        y_list = np.zeros((n_iter, n_test, n_obj))

        # initialize n_init solutions 

        x_init =  lhs(n_dim, n_init)
        if test_ins.startswith('zdt'):
            y_init = problem.evaluate(x_init)
            X = x_init
            Y = y_init
        else:
            y_init = problem.evaluate(torch.from_numpy(x_init).to(device))
            X = x_init
            Y = y_init.cpu().numpy()

        z = torch.zeros(n_obj).to(device)
        
        # n_iter batch selections 
        for i_iter in range(n_iter):
            print(f"Iteration:  {i_iter + 1 :03d}/{n_iter}  ||  Time: {(time.time() - start)/60:.2f} min")
            
            # intitialize the model and optimizer 
            psmodel = ParetoSetModel(n_dim, n_obj)
            psmodel.to(device)
                
            # optimizer
            optimizer = torch.optim.Adam(psmodel.parameters(), lr=lr)
            
            #  solution normalization
            transformation = StandardTransform([0,1])
            transformation.fit(X, Y)
            X_norm, Y_norm = transformation.do(X, Y) 

            # train GP surrogate model 
            surrogate_model = GaussianProcess(n_dim, n_obj, nu = 5)
            surrogate_model.fit(X_norm,Y_norm)
            
            z =  torch.min(torch.cat((z.reshape(1,n_obj),torch.from_numpy(Y_norm).to(device) - 0.1)), axis = 0).values.data
            
            # nondominated X, Y 
            nds = NonDominatedSorting()
            idx_nds = nds.do(Y_norm)
            
            X_nds = X_norm[idx_nds[0]]
            Y_nds = Y_norm[idx_nds[0]]

            alpha = np.ones(n_obj)
            pref_1 = np.random.dirichlet(alpha, 5)
            pref_vec_1  = torch.tensor(pref_1).to(device).float() + 0.0001


            # t_step Pareto Set Learning with Gaussian Process
            for t_step in range(n_steps):
                
                psmodel.train()

                
                # sample n_pref_update preferences
                alpha = np.ones(n_obj)
                pref = np.random.dirichlet(alpha, n_pref_update)
                pref_vec  = torch.tensor(pref).to(device).float() + 0.0001
                
                # get the current coressponding solutions
                x = psmodel(pref_vec)
                x_np = x.detach().cpu().numpy()

                # obtain the value/grad of mean/std for each obj
                mean = torch.from_numpy(surrogate_model.evaluate(x_np)['F']).to(device)
                mean_grad = torch.from_numpy(surrogate_model.evaluate(x_np, calc_gradient=True)['dF']).to(device)
                
                std = torch.from_numpy(surrogate_model.evaluate(x_np, std=True)['S']).to(device)
                std_grad = torch.from_numpy(surrogate_model.evaluate(x_np, std=True, calc_gradient=True)['dS']).to(device)
                
                # calculate the value/grad of tch decomposition with LCB
                # print('mean: ', mean)
                # print('std: ', std)
                value = mean - coef_lcb * std
                value_grad = mean_grad - coef_lcb * std_grad
                
                tch_idx = torch.argmax((1 / pref_vec) * (value - z), axis = 1)
                tch_idx_mat = [torch.arange(len(tch_idx)),tch_idx]
                tch_grad = (1 / pref_vec)[tch_idx_mat].view(n_pref_update,1) *  value_grad[tch_idx_mat] + 0.01 * torch.sum(value_grad, axis = 1) 

                tch_grad = tch_grad / torch.norm(tch_grad, dim = 1)[:, None]
                
                # gradient-based pareto set model update 
                optimizer.zero_grad()

                psmodel(pref_vec).backward(tch_grad)
                optimizer.step() 
                      
                
            print(f"   Training completed:   Time: {(time.time() - start)/60:.2f} min")
                
            # solutions selection on the learned Pareto set
            psmodel.eval()

            # sample n_candidate preferences
            pref = np.random.dirichlet(alpha, n_candidate)
            pref_vec  = torch.tensor(pref).to(device).float() + 0.0001

            # generate correponding solutions, get the predicted mean/std
            X_candidate = psmodel(pref_vec).to(torch.float64)
            # print(X_candidate)
            X_candidate_np = X_candidate.detach().cpu().numpy()
            Y_candidate_mean = surrogate_model.evaluate(X_candidate_np)['F']

            Y_candidata_std = surrogate_model.evaluate(X_candidate_np, std=True)['S']
            Y_candidate = Y_candidate_mean - coef_lcb * Y_candidata_std
            
            # optional TCH-based local Exploitation 
            if n_local > 0:
                X_candidate_tch = X_candidate_np
                z_candidate = z.cpu().numpy()
                pref_np = pref
                for j in range(n_local):
                    candidate_mean =  surrogate_model.evaluate(X_candidate_tch)['F']
                    candidate_mean_grad =  surrogate_model.evaluate(X_candidate_tch, calc_gradient=True)['dF']
                    
                    candidate_std = surrogate_model.evaluate(X_candidate_tch, std=True)['S']
                    candidate_std_grad = surrogate_model.evaluate(X_candidate_tch, std=True, calc_gradient=True)['dS']
                    
                    candidate_value = candidate_mean - coef_lcb * candidate_std
                    candidate_grad = candidate_mean_grad - coef_lcb * candidate_std_grad
                    
                    candidate_tch_idx = np.argmax((1 / pref_np) * (candidate_value - z_candidate), axis = 1)
                    candidate_tch_idx_mat = [np.arange(len(candidate_tch_idx)),list(candidate_tch_idx)]
                    
                    candidate_tch_grad = (1 / pref_np)[np.arange(len(candidate_tch_idx)),list(candidate_tch_idx)].reshape(n_candidate,1) * candidate_grad[np.arange(len(candidate_tch_idx)),list(candidate_tch_idx)] 
                    candidate_tch_grad +=  0.01 * np.sum(candidate_grad, axis = 1) 
                    
                    X_candidate_tch = X_candidate_tch - 0.01 * candidate_tch_grad
                    for i in range (n_dim):
                        X_candidate_tch[:,i] = np.clip(X_candidate_tch[:, i], a_min=problem.lbound[i].cpu(), a_max = problem.ubound[i].cpu())

                X_candidate_np = np.vstack([X_candidate_np, X_candidate_tch])
                
                Y_candidate_mean = surrogate_model.evaluate(X_candidate_np)['F']
                Y_candidata_std = surrogate_model.evaluate(X_candidate_np, std=True)['S']
                
                Y_candidate = Y_candidate_mean - coef_lcb * Y_candidata_std

            # greedy batch selection 
            best_subset_list = []
            Y_p = Y_nds 
            for b in range(n_sample):
                hv = HV(ref_point=np.max(np.vstack([Y_p,Y_candidate]), axis = 0))
                best_hv_value = 0
                best_subset = None
                
                for k in range(len(Y_candidate)):
                    Y_subset = Y_candidate[k]
                    Y_comb = np.vstack([Y_p,Y_subset])
                    hv_value_subset = hv.do(Y_comb)
                    if hv_value_subset > best_hv_value:
                        best_hv_value = hv_value_subset
                        best_subset = [k]
                        
                Y_p = np.vstack([Y_p,Y_candidate[best_subset].squeeze(0)])
                best_subset_list.append(best_subset)  
                
            best_subset_list = np.array(best_subset_list).T[0]
            
            # evaluate the selected n_sample solutions
            X_candidate = torch.tensor(X_candidate_np).to(device)
            X_new = X_candidate[best_subset_list]
            # print(X_new)
            if test_ins.startswith('zdt'):
                Y_new = problem.evaluate(X_new.detach().cpu().numpy())
            else:
                Y_new = problem.evaluate(X_new)

            X = np.vstack([X,X_new.detach().cpu().numpy()])
            if test_ins.startswith('zdt'):
                Y = np.vstack([Y,Y_new])
            else:
                Y = np.vstack([Y,Y_new.detach().cpu().numpy()])

            
            # check the current HV for evaluated solutions
            hv = HV(ref_point=np.array(ref_point))
            hv_value = hv(Y)
            hv_all_value[run_iter, i_iter] = hv_value
            
            pref_vec  = torch.Tensor(pref_vec_test).to(device).float()

            x = psmodel(pref_vec)
            
            front_list[i_iter] = surrogate_model.evaluate(x.detach().cpu().numpy())['F']
            x_list[i_iter] = x.detach().cpu().numpy()
            if test_ins.startswith('zdt'):
                y_list[i_iter] = problem.evaluate(x.detach().cpu().numpy())
            else:
                y_list[i_iter] = problem.evaluate(x.to(device)).detach().cpu().numpy()
        
        
            print(f"   Testing completed:    Time: {(time.time() - start)/60:.2f} min")
        
            # store the final performance
            hv_list[test_ins] = hv_all_value
        
            print("***********************************************")

            np.save(f"result/PSL_MOBO/batch_{n_sample}/logs_{coef_lcb}_{test_ins}_{n_obj}_{n_dim}_{suffix}/evaluation_{test_ins}_X_{n_dim}{suffix}_{run_iter}", X)
            np.save(f"result/PSL_MOBO/batch_{n_sample}/logs_{coef_lcb}_{test_ins}_{n_obj}_{n_dim}_{suffix}/evaluation_{test_ins}_Y_{n_dim}{suffix}_{run_iter}", Y)
            
            np.save(f"result/PSL_MOBO/batch_{n_sample}/logs_{coef_lcb}_{test_ins}_{n_obj}_{n_dim}_{suffix}/front_{test_ins}_{n_dim}{suffix}_{run_iter}", front_list)
            np.save(f"result/PSL_MOBO/batch_{n_sample}/logs_{coef_lcb}_{test_ins}_{n_obj}_{n_dim}_{suffix}/x_{test_ins}_{n_dim}{suffix}_{run_iter}", x_list)
            np.save(f"result/PSL_MOBO/batch_{n_sample}/logs_{coef_lcb}_{test_ins}_{n_obj}_{n_dim}_{suffix}/y_{test_ins}_{n_dim}{suffix}_{run_iter}", y_list)
            print("hv", "{:.2e}".format(np.mean(hv_value)))
            print("***")
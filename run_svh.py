import numpy as np
import torch
import pickle

from problem import get_problem
from pymoo.problems import get_problem as get_problem_pymoo

from lhs import lhs
from pymoo.indicators.hv import HV
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting

# from mobo.surrogate_model import GaussianProcess
from mobo.gp_model.gaussian_process import GaussianProcess
from mobo.transformation import StandardTransform

from model import ParetoSetModel
from SVGD.moosvgd import get_gradient
from torch.autograd import Variable
import random
from mobo.utils import set_seed
import os
# from partition import  sampling_vector_evenly
from partitioning import sampling_vector_randomly, sampling_vector_evenly
import time
import torch.nn as nn 

# Set seed
set_seed(42)
dic = {'zdt1':[0.9994, 6.0576],
        'zdt2':[0.9994, 6.8960],
        'zdt3':[0.9994, 6.0571],
        'RE21':[2886.3695604236013, 0.039999999999998245],
       'RE32':[ 37.7831517014 , 17561.6 ,425062976.628],
       'RE33':[5.3067 , 3.12833430979 , 25.0 ],
       'RE36':[5.931, 56.0, 0.355720675227],
       'RE37':[0.98949120096, 0.956587924661 , 0.987530948586],
       'RE41':[39.2905121788, 4.42725, 13.09138125 ,9.49401929991],
       'RE42': [-844.714092162, 13827.1384409, 5707.50786547, 3207.0456123],
       'VLMOP2': [1.1,1.1],
       "F2": [1.1,1.1]
       }

ins_list = ['zdt1', 'zdt2', 'zdt3', 'VLMOP2','F2', 'RE21', 'RE32', 'RE33', 'RE36', 'RE37', 'RE41', 'RE42']
# number of independent runs
n_run = 5
# number of initialized solutions
n_init = 20
# number of iterations, and batch size per iteration
n_iter = 20
batch_size = 5
# PSL 
# number of learning steps
n_steps = 1000
# coefficient of LCB
coef_lcb = 0.1
# number of sampled candidates on the approxiamte Pareto front
n_candidate = 1000

local_kernel= True

# device
def get_device(no_cuda=False, gpus=None):
    return torch.device(f"cuda:{gpus}" if torch.cuda.is_available() and not no_cuda else "cpu")
device=get_device(gpus ='0')

c_ = 1
alpha_ = 0.001
lr_ = 1e-3

print(device)
print(f'c: {c_}, alpha: {alpha_}, lr: {lr_}')
print('step: ',n_steps)
print('batch_sizes', batch_size)

# -----------------------------------------------------------------------------
start = time.time()
hv_list = {}
for test_ins in ins_list:
    print(test_ins)

    if test_ins in ['VLMOP2','F2']:
        n_dim = 6
    elif test_ins in ['zdt1','zdt2','zdt3']:
        n_dim = 30

    hv_all_value = np.zeros([n_iter])
    # problem = get_problem(test_ins, n_dim = n_dim)
    if test_ins.startswith('zdt'):
        problem = get_problem_pymoo(test_ins, n_var=n_dim)
        n_dim = problem.n_var
        n_obj = problem.n_obj
    else:
        problem = get_problem(test_ins)
        n_dim = problem.n_dim
        n_obj = problem.n_obj

    ref_point = dic[test_ins]

    name = f'c_{c_}_alpha_{alpha_}_lr_{lr_}_steps_{n_steps}_n_cand{n_candidate}_{n_obj}_{n_dim}_batch_sizes_{batch_size}'
    suffix=f"_stein_local"
    suffix_dir = ""
    
    if not os.path.exists(f"result/SVH_PSL/batch_{batch_size}/logs_{test_ins}{suffix}_{name}"):
            os.makedirs(f"result/SVH_PSL/batch_{batch_size}/logs_{test_ins}{suffix}_{name}")
    folder_path_rs = f'result/SVH_PSL/batch_{batch_size}'
    
    n_pref_update = 10

    n_test = 100*n_obj
    pref_vec_test = sampling_vector_evenly(n_obj, n_test)
    
    print(f"Problem: {test_ins}\nN dim: {n_dim} \nN objective: {n_obj} \nLogs dir: logs_{test_ins}{suffix}/\nRun: {suffix}")
    print("***********************************************")
    
    for run_iter in range(n_run):
        front_list,gp_list, x_list, y_list = {}, {}, {},{}
        front_list = np.zeros((n_iter, n_test, n_obj))
        x_list = np.zeros((n_iter, n_test, n_dim))
        y_list = np.zeros((n_iter, n_test, n_obj))
        gp_list = np.zeros((n_iter, n_test, n_obj))
        x_init = lhs(n_dim, n_init)
        if test_ins.startswith('zdt') :
            y_init = problem.evaluate(x_init)
            X = x_init
            Y = y_init
        else:
            y_init = problem.evaluate(torch.from_numpy(x_init).to(device))
            X = x_init
            Y = y_init.to(torch.float64).cpu().numpy()
         
        z = torch.zeros(n_obj).to(device)

        for i_iter in range(n_iter):
            print(f"Iteration:  {i_iter + 1 :03d}/{n_iter}  ||  Time: {(time.time() - start)/60:.2f} min")

            # create pareto set model
            psmodel = ParetoSetModel(n_dim, n_obj)
            psmodel.to(device)

            # get parameters of the Pareto set model for learning SVH
            params = list(psmodel.parameters())

            # optimizer
            optimizer = torch.optim.Adam(psmodel.parameters(), lr=lr_)
            if test_ins in ['zdt1','zdt2','zdt3']:
                X_norm = X
                Y_norm = Y
            else:
                # solution normalization
                transformation = StandardTransform([0,1])
                transformation.fit(X, Y)
                X_norm,Y_norm = transformation.do(X,Y)

            nds = NonDominatedSorting()
            idx_nds = nds.do(Y_norm)
            X_norm = torch.tensor(X_norm).to(device)
            Y_norm = torch.tensor(Y_norm).to(device)
            
            # train GP surrogate model 
            surrogate_model = GaussianProcess(X_norm, Y_norm,300, n_dim, n_obj, nu = 2.5, device = device)
            surrogate_model.fit(X_norm,Y_norm)
            
            z =  torch.min(torch.cat((z.reshape(1,n_obj),Y_norm - 0.1)), axis = 0).values.data
            
            X_nds = X_norm[idx_nds[0]]
            Y_nds = Y_norm[idx_nds[0]]
            
            hv_max = 0.0

            for t_step in range(n_steps):
                psmodel.train()
                
                # sample n_pref_update preferences
                alpha = np.ones(n_obj)
                pref = np.random.dirichlet(alpha,n_pref_update)
                pref_vec  = torch.tensor(pref).to(device).float() + 0.0001
                # get the current coressponding solutions
                x = psmodel(pref_vec)

                x_np = x.detach().cpu().numpy()
                mean, std = surrogate_model.predict(x)
                mean = torch.stack(mean)
                std = torch.stack(std)

                value = mean - coef_lcb * std

                value = value.T
                idx_che = None
                loss,idx_che = torch.max((value - z)* pref_vec, dim=1)
                

                grad = []
                for idx,loss_ in enumerate(loss):
                    grad_ = []
                    loss_.backward(retain_graph=True)
                    for i,param in enumerate(params):
                        grad_.append(Variable(param.grad.data.clone().flatten(), requires_grad=False))
                        param.grad.zero_()
                    grad.append(torch.cat(grad_, dim=0))

                grad = torch.stack(grad, dim=0)
                grad_1 = torch.nn.functional.normalize(grad, dim=0)
                grad_2 = None

                # gradient-based pareto set model update 
                optimizer.zero_grad()
                grad = get_gradient(grad_1, params, value, alpha_, c_ ,grad_2,pref_vec,idx_che, local_kernel=local_kernel)
                grad = torch.sum(grad, dim=0)

                idx = 0
                for i,param in enumerate(params):
                    param.grad.data = grad[idx:idx+param.numel()].reshape(param.shape)
                    idx +=param.numel()
                optimizer.step()


                # select checkpoint weights Pareto set model
                Y_p = Y_nds.detach().cpu().numpy()
                pref_vec  = torch.Tensor(pref_vec_test).to(device).float()
                x = psmodel(pref_vec)
                Y_candidate_mean__, Y_candidata_std__ = surrogate_model.predict(x)
                Y_candidate_mean__ = torch.stack(Y_candidate_mean__)
                Y_candidata_std__ = torch.stack(Y_candidata_std__)
                Y_ = Y_candidate_mean__ - coef_lcb*Y_candidata_std__
                Y_ = Y_.T.detach().cpu().numpy()
                hv = HV(ref_point=np.max(np.vstack([Y_p,Y_]), axis = 0))
                hv_value_eval_psmodel = hv(Y_)


                if not os.path.exists('weights'):
                    os.makedirs('weights')
                if hv_max <= hv_value_eval_psmodel:
                    hv_max = hv_value_eval_psmodel
                    torch.save(psmodel,f'weights/{test_ins}{suffix}_{name}.pt')


            print(f"   Training completed:   Time: {(time.time() - start)/60:.2f} min")
            if hv_max >0:
                psmodel = torch.load(f'weights/{test_ins}{suffix}_{name}.pt', map_location=device)
            psmodel.eval()

            # solutions selection on the learned Pareto set
            # sample n_candidate preferences
            alpha = np.ones(n_obj)
            pref = np.random.dirichlet(alpha,n_candidate)
            pref  = torch.tensor(pref).to(device).float() + 0.0001

            # generate correponding solutions, get the predicted mean/std
            X_candidate = psmodel(pref).to(torch.float32)
            # print(X_candidate)
            X_candidate_np = X_candidate.detach().cpu().numpy()
            Y_candidate_mean, Y_candidata_std = surrogate_model.predict(X_candidate)

            Y_candidate_mean = torch.stack(Y_candidate_mean)
            Y_candidata_std = torch.stack(Y_candidata_std)
            Y_candidate = Y_candidate_mean - coef_lcb * Y_candidata_std

            Y_candidate = Y_candidate.T.detach().cpu().numpy()
            
            # greedy batch selection 
            best_idx = []
            best_subset_list = []
            Y_p = Y_nds.detach().cpu().numpy()
            # print(Y_candidate)
            for b in range(batch_size):
                hv = HV(ref_point=np.max(np.vstack([Y_p,Y_candidate]), axis = 0))
                best_hv_value = 0
                best_subset = None
                
                for k in range(len(Y_candidate)):
                    # if k not in best_idx:
                        Y_subset = Y_candidate[k]
                        Y_comb = np.vstack([Y_p,Y_subset])
                        hv_value_subset = hv(Y_comb)
                        if hv_value_subset > best_hv_value:
                            best_hv_value = hv_value_subset
                            best_subset = [k]  

                Y_p = np.vstack([Y_p,Y_candidate[best_subset]])
                best_subset_list.append(best_subset)  
                
            best_subset_list = np.array(best_subset_list).T[0]
            
            # evaluate the selected batch_size solutions
            X_candidate = torch.tensor(X_candidate_np).to(device)
            X_new = X_candidate[best_subset_list]
            if test_ins.startswith('zdt'):
                Y_new = problem.evaluate(X_new.detach().cpu().numpy())
            else:
                Y_new = problem.evaluate(X_new)
            
            # update the set of evaluated solutions (X,Y)
            X = np.vstack([X,X_new.detach().cpu().numpy()])

            if test_ins.startswith('zdt'):
                Y = np.vstack([Y,Y_new])
            else:
                Y = np.vstack([Y,Y_new.detach().cpu().numpy()])
            
            # check the current HV for evaluated solutions
            hv = HV(ref_point=np.array(ref_point))
            hv_value = hv(Y)
            hv_all_value[i_iter] = hv_value

            pref_vec  = torch.Tensor(pref_vec_test).to(device).float()

            x = psmodel(pref_vec)
       
            Y_candidate_mean__, Y_candidata_std__ = surrogate_model.predict(x)
            Y_candidate_mean__ = torch.stack(Y_candidate_mean__)
            Y_candidata_std__ = torch.stack(Y_candidata_std__)
            Y_candidate__ = Y_candidate_mean__ - coef_lcb*Y_candidata_std__
            

            front_list[i_iter] = Y_candidate_mean__.T.detach().cpu().numpy()
            gp_list[i_iter] = Y_candidate__.T.detach().cpu().numpy()
            x_list[i_iter] = x.detach().cpu().numpy()
    
            if test_ins.startswith('zdt'):
                y_list[i_iter] = problem.evaluate(x.detach().cpu().numpy())
            else:
                y_list[i_iter] = problem.evaluate(x.to(device)).detach().cpu().numpy()

        
            print(f"   Testing completed:    Time: {(time.time() - start)/60:.2f} min")
        
            # store the final performance
            hv_list[test_ins] = hv_all_value
        
            print("***********************************************")

            
            np.save(f"{folder_path_rs}/logs_{test_ins}{suffix}_{name}/evaluation_{test_ins}_X_{n_dim}{suffix}_{run_iter}", X)
            np.save(f"{folder_path_rs}/logs_{test_ins}{suffix}_{name}/evaluation_{test_ins}_Y_{n_dim}{suffix}_{run_iter}", Y)
            
            np.save(f"{folder_path_rs}/logs_{test_ins}{suffix}_{name}/front_{test_ins}_{n_dim}{suffix}_{run_iter}", front_list)
            np.save(f"{folder_path_rs}/logs_{test_ins}{suffix}_{name}/x_{test_ins}_{n_dim}{suffix}_{run_iter}", x_list)
            np.save(f"{folder_path_rs}/logs_{test_ins}{suffix}_{name}/y_{test_ins}_{n_dim}{suffix}_{run_iter}", y_list)
            
            
            print("hv", "{:.2e}".format(np.mean(hv_value)))
            print("***")


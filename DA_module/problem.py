import torch
import numpy as np


def get(name, *args, **kwargs):
    name = name.lower()

    PROBLEM = {
        'vlmop2': VLMOP2,
        're21':RE21,
        're32': RE32,
        're33': RE33,
        're36': RE36,
        're37': RE37,
        're41': RE41
    }

    if name not in PROBLEM:
        raise Exception("Problem not found.")

    return PROBLEM[name](*args, **kwargs)



class RE21():
    def __init__(self, n_var=4):
        F = 10.0
        sigma = 10.0
        tmp_val = F / sigma

        self.n_var = n_var
        self.n_obj = 2
        self.lbound = torch.tensor([tmp_val, np.sqrt(2.0) * tmp_val, np.sqrt(2.0) * tmp_val, tmp_val]).float()
        self.ubound = torch.ones(n_var).float() * 3 * tmp_val
        self.nadir_point = [2886.3695604236013, 0.039999999999998245]

    def evaluate(self, x):
        x = torch.from_numpy(x).to('cuda')
        F = 10.0
        E = 2.0 * 1e5
        L = 200.0

        if x.device.type == 'cuda':
            self.lbound = self.lbound.cuda()
            self.ubound = self.ubound.cuda()

        x = x * (self.ubound - self.lbound) + self.lbound

        f1 = L * ((2 * x[:, 0]) + np.sqrt(2.0) * x[:, 1] + torch.sqrt(x[:, 2]) + x[:, 3])
        f2 = ((F * L) / E) * (
                    (2.0 / x[:, 0]) + (2.0 * np.sqrt(2.0) / x[:, 1]) - (2.0 * np.sqrt(2.0) / x[:, 2]) + (2.0 / x[:, 3]))

        f1 = f1
        f2 = f2

        objs = torch.stack([f1, f2]).T
        return objs.cpu().numpy()


class RE33():
    def __init__(self, n_var=4):
        self.n_var = n_var
        self.n_obj = 3
        self.lbound = torch.tensor([55, 75, 1000, 11]).float()
        self.ubound = torch.tensor([80, 110, 3000, 20]).float()
        self.nadir_point = [5.3067, 3.12833430979, 25.0]

    def evaluate(self, x):
        x = torch.from_numpy(x).to('cuda')
        if x.device.type == 'cuda':
            self.lbound = self.lbound.cuda()
            self.ubound = self.ubound.cuda()

        x = x * (self.ubound - self.lbound) + self.lbound

        x1 = x[:, 0]
        x2 = x[:, 1]
        x3 = x[:, 2]
        x4 = x[:, 3]

        # First original objective function
        f1 = 4.9 * 1e-5 * (x2 * x2 - x1 * x1) * (x4 - 1.0)
        # Second original objective function
        f2 = ((9.82 * 1e6) * (x2 * x2 - x1 * x1)) / (x3 * x4 * (x2 * x2 * x2 - x1 * x1 * x1))

        # Reformulated objective functions
        g1 = (x2 - x1) - 20.0
        g2 = 0.4 - (x3 / (3.14 * (x2 * x2 - x1 * x1)))
        g3 = 1.0 - (2.22 * 1e-3 * x3 * (x2 * x2 * x2 - x1 * x1 * x1)) / torch.pow((x2 * x2 - x1 * x1), 2)
        g4 = (2.66 * 1e-2 * x3 * x4 * (x2 * x2 * x2 - x1 * x1 * x1)) / (x2 * x2 - x1 * x1) - 900.0

        g = torch.stack([g1, g2, g3, g4])
        z = torch.zeros(g.shape).float().cuda().to(torch.float64)
        g = torch.where(g < 0, -g, z)

        f3 = torch.sum(g, axis=0).float()

        objs = torch.stack([f1, f2, f3]).T

        return objs.cpu().numpy()


class RE36():
    def __init__(self, n_var=4):
        self.n_var = n_var
        self.n_obj = 3
        self.lbound = torch.tensor([12, 12, 12, 12]).float()
        self.ubound = torch.tensor([60, 60, 60, 60]).float()
        self.nadir_point = [5.931, 56.0, 0.355720675227]

    def evaluate(self, x):
        x = torch.from_numpy(x).to('cuda')
        if x.device.type == 'cuda':
            self.lbound = self.lbound.cuda()
            self.ubound = self.ubound.cuda()

        x = x * (self.ubound - self.lbound) + self.lbound

        x1 = x[:, 0]
        x2 = x[:, 1]
        x3 = x[:, 2]
        x4 = x[:, 3]

        # First original objective function
        f1 = torch.abs(6.931 - ((x3 / x1) * (x4 / x2)))
        # Second original objective function (the maximum value among the four variables)
        l = torch.stack([x1, x2, x3, x4])
        f2 = torch.max(l, dim=0)[0]

        g1 = 0.5 - (f1 / 6.931)

        g = torch.stack([g1])
        z = torch.zeros(g.shape).float().cuda().to(torch.float64)
        g = torch.where(g < 0, -g, z)
        f3 = g[0]

        objs = torch.stack([f1, f2, f3]).T

        return objs.cpu().numpy()


class RE37():
    def __init__(self, n_var=4):
        self.n_var = n_var
        self.n_obj = 3
        self.lbound = torch.tensor([0, 0, 0, 0]).float()
        self.ubound = torch.tensor([1, 1, 1, 1]).float()
        self.nadir_point = [0.98949120096, 0.956587924661, 0.987530948586]

    def evaluate(self, x):
        x = torch.from_numpy(x).to('cuda')
        if x.device.type == 'cuda':
            self.lbound = self.lbound.cuda()
            self.ubound = self.ubound.cuda()

        x = x * (self.ubound - self.lbound) + self.lbound

        xAlpha = x[:, 0]
        xHA = x[:, 1]
        xOA = x[:, 2]
        xOPTT = x[:, 3]

        # f1 (TF_max)
        f1 = 0.692 + (0.477 * xAlpha) - (0.687 * xHA) - (0.080 * xOA) - (0.0650 * xOPTT) - (0.167 * xAlpha * xAlpha) - (
                    0.0129 * xHA * xAlpha) + (0.0796 * xHA * xHA) - (0.0634 * xOA * xAlpha) - (0.0257 * xOA * xHA) + (
                         0.0877 * xOA * xOA) - (0.0521 * xOPTT * xAlpha) + (0.00156 * xOPTT * xHA) + (
                         0.00198 * xOPTT * xOA) + (0.0184 * xOPTT * xOPTT)
        # f2 (X_cc)
        f2 = 0.153 - (0.322 * xAlpha) + (0.396 * xHA) + (0.424 * xOA) + (0.0226 * xOPTT) + (0.175 * xAlpha * xAlpha) + (
                    0.0185 * xHA * xAlpha) - (0.0701 * xHA * xHA) - (0.251 * xOA * xAlpha) + (0.179 * xOA * xHA) + (
                         0.0150 * xOA * xOA) + (0.0134 * xOPTT * xAlpha) + (0.0296 * xOPTT * xHA) + (
                         0.0752 * xOPTT * xOA) + (0.0192 * xOPTT * xOPTT)
        # f3 (TT_max)
        f3 = 0.370 - (0.205 * xAlpha) + (0.0307 * xHA) + (0.108 * xOA) + (1.019 * xOPTT) - (0.135 * xAlpha * xAlpha) + (
                    0.0141 * xHA * xAlpha) + (0.0998 * xHA * xHA) + (0.208 * xOA * xAlpha) - (0.0301 * xOA * xHA) - (
                         0.226 * xOA * xOA) + (0.353 * xOPTT * xAlpha) - (0.0497 * xOPTT * xOA) - (
                         0.423 * xOPTT * xOPTT) + (0.202 * xHA * xAlpha * xAlpha) - (0.281 * xOA * xAlpha * xAlpha) - (
                         0.342 * xHA * xHA * xAlpha) - (0.245 * xHA * xHA * xOA) + (0.281 * xOA * xOA * xHA) - (
                         0.184 * xOPTT * xOPTT * xAlpha) - (0.281 * xHA * xAlpha * xOA)

        objs = torch.stack([f1, f2, f3]).T

        return objs.cpu().numpy()
        
def div(x1, x2):

  return np.divide(x1, x2, out=np.zeros(np.broadcast(x1, x2).shape), where=(x2 != 0))


    
class RE32():
    def __init__(self, n_var = 4):
        
      
        self.n_var = 4
        self.n_obj = 3
        self.lbound = torch.tensor([0.125, 0.1, 0.1, 0.125]).float()
        self.ubound = torch.tensor([5, 10, 10, 5]).float()
        self.nadir_point = [0.98949120096, 0.956587924661, 0.987530948586]
    def evaluate(self, x):
        x = torch.from_numpy(x).to('cuda')
        if x.device.type == 'cuda':        
            self.lbound = self.lbound.cuda()
            self.ubound = self.ubound.cuda()
        
        x = x * (self.ubound - self.lbound) + self.lbound
        x1, x2, x3, x4 = x[:, 0], x[:, 1], x[:, 2], x[:, 3]
        
        P = 6000
        L = 14
        E = 30 * 1e6
        G = 12 * 1e6
        tauMax = 13600
        sigmaMax = 30000
        x1 = x1.cpu().numpy()
        x2 = x2.cpu().numpy()
        x3 = x3.cpu().numpy()
        x4 = x4.cpu().numpy()
        f1 = (1.10471 * x1 * x1 * x2) + (0.04811 * x3 * x4) * (14.0 + x2)
        f2 = div(4 * P * L * L * L, E * x4 * x3 * x3 * x3)

        M = P * (L + (x2 / 2))
        tmpVar = ((x2 * x2) / 4.0) + np.power((x1 + x3) / 2.0, 2)
        R = np.sqrt(tmpVar)
        tmpVar = ((x2 * x2) / 12.0) + np.power((x1 + x3) / 2.0, 2)
        J = 2 * np.sqrt(2) * x1 * x2 * tmpVar

        tauDashDash = div(M * R, J)
        tauDash = div(P, np.sqrt(2) * x1 * x2)
        tmpVar = tauDash * tauDash + div((2 * tauDash * tauDashDash * x2), (2 * R)) + (tauDashDash * tauDashDash)
        tau = np.sqrt(tmpVar)
        sigma = div(6 * P * L, x4 * x3 * x3)
        tmpVar = 4.013 * E * np.sqrt((x3 * x3 * x4 * x4 * x4 * x4 * x4 * x4) / 36.0) / (L * L)
        tmpVar2 = (x3 / (2 * L)) * np.sqrt(E / (4 * G))
        PC = tmpVar * (1 - tmpVar2)

        g = np.column_stack([
            tauMax - tau,
            sigmaMax - sigma,
            x4 - x1,
            PC - P
        ])

        g[g >= 0] = 0
        g[g < 0] = -g[g < 0]

        f3 = np.sum(g, axis=1)
        f1 = torch.tensor(f1).to('cuda').float()
        f2 = torch.tensor(f2).to('cuda').float()
        f3 = torch.tensor(f3).to('cuda').float()
        objs = torch.stack([f1, f2, f3]).T
        return objs.cpu().numpy()


class RE41():
    def __init__(self):
        self.problem_name = 'RE41'
        self.n_obj = 4
        self.n_var = 7
        self.n_constraints = 0
        self.n_original_constraints = 10

        self.lbound = torch.zeros(self.n_var).float()
        self.ubound = torch.zeros(self.n_var).float()
        self.lbound[0] = 0.5
        self.lbound[1] = 0.45 
        self.lbound[2] = 0.5 
        self.lbound[3] = 0.5 
        self.lbound[4] = 0.875 
        self.lbound[5] = 0.4 
        self.lbound[6] = 0.4 
        self.ubound[0] = 1.5
        self.ubound[1] = 1.35
        self.ubound[2] = 1.5
        self.ubound[3] = 1.5
        self.ubound[4] = 2.625
        self.ubound[5] = 1.2
        self.ubound[6] = 1.2
                
    def evaluate(self, x):
        x = torch.from_numpy(x).to('cuda')
        if x.device.type == 'cuda': 
            self.lbound = self.lbound.to(x.device)
            self.ubound = self.ubound.to(x.device)
        x = x * (self.ubound - self.lbound) + self.lbound
        f = torch.zeros((x.shape[0],self.n_obj)).to(x.device).float()
        g = torch.zeros((x.shape[0],self.n_original_constraints)).to(x.device).float()

        x1 = x[:,0]
        x2 = x[:,1]
        x3 = x[:,2]
        x4 = x[:,3]
        x5 = x[:,4]
        x6 = x[:,5]
        x7 = x[:,6]

        # First original objective function
        f[:,0] = 1.98 + 4.9 * x1 + 6.67 * x2 + 6.98 * x3 + 4.01 * x4 + 1.78 * x5 + 0.00001 * x6 + 2.73 * x7
        # Second original objective function
        f[:,1] = 4.72 - 0.5 * x4 - 0.19 * x2 * x3
        # Third original objective function
        Vmbp = 10.58 - 0.674 * x1 * x2 - 0.67275 * x2
        Vfd = 16.45 - 0.489 * x3 * x7 - 0.843 * x5 * x6
        f[:,2] = 0.5 * (Vmbp + Vfd)

        # Constraint functions
        g[:,0] = 1 -(1.16 - 0.3717 * x2 * x4 - 0.0092928 * x3)
        g[:,1] = 0.32 -(0.261 - 0.0159 * x1 * x2 - 0.06486 * x1 -  0.019 * x2 * x7 + 0.0144 * x3 * x5 + 0.0154464 * x6)
        g[:,2] = 0.32 -(0.214 + 0.00817 * x5 - 0.045195 * x1 - 0.0135168 * x1 + 0.03099 * x2 * x6 - 0.018 * x2 * x7  + 0.007176 * x3 + 0.023232 * x3 - 0.00364 * x5 * x6 - 0.018 * x2 * x2)
        g[:,3] = 0.32 -(0.74 - 0.61 * x2 - 0.031296 * x3 - 0.031872 * x7 + 0.227 * x2 * x2)
        g[:,4] = 32 -(28.98 + 3.818 * x3 - 4.2 * x1 * x2 + 1.27296 * x6 - 2.68065 * x7)
        g[:,5] = 32 -(33.86 + 2.95 * x3 - 5.057 * x1 * x2 - 3.795 * x2 - 3.4431 * x7 + 1.45728)
        g[:,6] =  32 -(46.36 - 9.9 * x2 - 4.4505 * x1)
        g[:,7] =  4 - f[:,1]
        g[:,8] =  9.9 - Vmbp
        g[:,9] =  15.7 - Vfd

        g = torch.where(g < 0, -g, 0)                
        f[:,3] = g[:,0] + g[:,1] + g[:,2] + g[:,3] + g[:,4] + g[:,5] + g[:,6] + g[:,7] + g[:,8] + g[:,9]  

        return f.cpu().numpy()


class VLMOP2():
    def __init__(self, n_var = 6):
        self.n_var = n_var
        self.n_obj = 2
        self.lbound = torch.tensor([-2.0, -2.0, -2.0, -2.0, -2.0, -2.0]).float()
        self.ubound = torch.tensor([2.0, 2.0,2.0, 2.0,2.0, 2.0]).float()
        # self.lbound = torch.tensor([-1]*n_var).float()
        # self.ubound = torch.tensor([1]*n_var).float()
        self.nadir_point = [1, 1]
        self.bound = 2
       
    def evaluate(self, x):
        x = torch.from_numpy(x).to('cuda')
        n = x.shape[1]
        
        if x.device.type == 'cuda':        
            self.lbound = self.lbound.to(x.device)
            self.ubound = self.ubound.to(x.device)
        
        x = x * (self.ubound - self.lbound) + self.lbound
        
        f1 = 1 - torch.exp(-torch.sum((x - 1 / np.sqrt(n))**2, axis = 1))
        f2 = 1 - torch.exp(-torch.sum((x + 1 / np.sqrt(n))**2, axis = 1))
     
        objs = torch.stack([f1,f2]).T
        
        return objs.cpu().numpy()
    
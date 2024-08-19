import numpy as np
from mobo.gp_model.base_surrogate_model import SurrogateModel
from mobo.gp_model.models import GP_Model


class GaussianProcess(SurrogateModel):
    def __init__(self,X,y, training_iters, n_var, n_obj, nu = 2.5, device = None, **kwargs):
        super().__init__(n_var, n_obj)
        self.nu = nu
        self.gps = []
        self.X = X
        self.y = y
        self.device = device
        self.training_iters = training_iters
        # print(n_obj)
        for i in range(n_obj):
            # if i ==1:
            #     gp = GP_Model(self.X, self.y[:,i],nu =self.nu, n_restarts=2, device = self.device)
            # else:
            # print(self.X.shape)
            # print(self.y[:,i].shape)
            # print(i)
            gp = GP_Model(self.X, self.y[:,i],training_iters= self.training_iters, nu =self.nu, device = self.device)
            self.gps.append(gp)
    
    def fit(self, X,y):
        for i,gp in enumerate(self.gps):
            gp.fit(X, y[:,i])
    
    def predict(self, X):
        F, S= [], []
        for gp in self.gps:
            mean_ = gp.predict(X)
            std_ = gp.variance(X)

            F.append(mean_)
            S.append(std_)
        
        return F, S

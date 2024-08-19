from abc import ABC, abstractmethod

'''
Surrogate model that predicts the performance of given design variables
'''

class SurrogateModel(ABC):
    '''
    Base class of surrogate model
    '''
    def __init__(self, n_var, n_obj):
        self.n_var = n_var
        self.n_obj = n_obj

    @abstractmethod
    def fit(self, X, Y):
        '''
        Fit the surrogate model from data (X, Y)
        '''
        pass

    @abstractmethod
    def predict(self, X):
        '''
        predict mean and standard deviation
        '''
        pass

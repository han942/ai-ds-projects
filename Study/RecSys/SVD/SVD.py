import numpy as np
import pandas as pd
from scipy.linalg import svd

class SVD():
    def __init__(self,R,k,epochs):
        self.R = R # m*n
        self.k = k
        self.epochs = epochs

         # Scipy's SVD decomposition
        self.U,self.s,self.V = svd(self.R)
        self.sigma = np.zeros(self.R.shape)
        self.sigma[:len(self.s),:len(self.s)] = np.diag(self.s)
        pass
        
    def fit(self):
        self.U_k = self.U[:,:self.k]
        self.sigma_k = self.sigma[:self.k,:self.k]
        self.V_k = self.V[:self.k,:]   
        pass

    def predict(self):
        self.R_pred = self.U_k @ self.sigma_k @ self.V_k
        return self.R_pred
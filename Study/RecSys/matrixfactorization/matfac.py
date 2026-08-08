import numpy as np
import pandas as pd
from sklearn.metrics import root_mean_squared_error

class  MatrixFactorization():
    def __init__(self,k,lr,reg_param,epochs):
        self.k = k
        self.lr = lr
        self.reg_param = reg_param
        self.epochs = epochs
        pass

    def fit(self,train):
        R = pd.pivot_table(train,values='rating',index='user',columns='item')
        R = R.fillna(0)
        self.R_df = R
        self.n_users,self.n_items = R.shape

        self.obs_rows,self.obs_cols = np.nonzero(R) #Index conversion for observed data in UI matrix(R)
        self.obs_ind = list(zip(self.obs_rows,self.obs_cols))

        self.train_user,self.train_item = train['user'].unique(),train['item'].unique()
        
        self.R = np.array(R)    # u,i is index for array
        #P,Q random initialization
        self.P = np.random.random(size=(self.n_users,self.k)) *(6/self.k)
        self.Q = np.random.random(size=(self.n_items,self.k)) *(6/self.k)

        #Bias
        self.b_u = np.zeros(self.n_users)
        self.b_i = np.zeros(self.n_items)

        self.global_mean = R[R>0].mean().mean()
        

        for n in range(self.epochs):
            for u,i in self.obs_ind: #Index-level operating
                    if self.R[u,i] == 0:
                        pass
                    else:
                        e = self.R[u,i] - (np.dot(self.P[u,:],self.Q[i,:].T) + self.b_u[u] + self.b_i[i] +self.global_mean)  #error
                        #user-update
                        self.P[u,:] = self.P[u,:] + self.lr * (e* self.Q[i,:] - self.reg_param*self.P[u,:])
                        #item-update
                        self.Q[i,:] = self.Q[i,:] + self.lr * (e* self.P[u,:] - self.reg_param*self.Q[i,:])

                        #Bias-update
                        self.b_u[u] = self.b_u[u] + self.lr * (e - self.reg_param*self.b_u[u])
                        self.b_i[i] = self.b_i[i] + self.lr * (e - self.reg_param*self.b_i[i])
            if n % 10 == 0:
                R_pred = np.dot(self.P,self.Q.T) + self.b_u[:,np.newaxis] + self.b_i[np.newaxis,:] + self.global_mean
                bias_term =  np.sum(np.square(self.b_u)) + np.sum(np.square(self.b_i))
                # Loss Function
                loss = (np.sum(np.square(self.R - R_pred)) + self.reg_param*(np.sum(np.square(self.P)) + np.sum(np.square(self.Q))+bias_term)) / len(self.obs_ind)
                print(f'Epoch : {n} , Loss : {loss:4f} , Rooted Loss: {np.sqrt(loss):.2f}')
        return self.P,self.Q,self.b_u,self.b_i

    def predict(self,test,exclude_unknowns=True):
        P_df = pd.DataFrame(self.P,index=self.R_df.index)
        Q_df = pd.DataFrame(self.Q,index=self.R_df.columns)
        bu_df = pd.DataFrame(self.b_u,index=self.R_df.index)
        bi_df = pd.DataFrame(self.b_i,index=self.R_df.columns)
        

        if exclude_unknowns == True: 
            test_filtered = test[(test['user'].isin(self.train_user)) & (test['item'].isin(self.train_item))]
            uilist_test = list(zip(test_filtered['user'],test_filtered['item']))
            prediction = test_filtered.copy()
            pred = []
            for val in uilist_test:
                pred.append(float(np.dot(P_df.loc[val[0]],Q_df.loc[val[1]].T) + bu_df.loc[val[0]].values + bi_df.loc[val[1]].values + self.global_mean)) # P,Q is array
            prediction['rating'] = pred
        
        # else:
        #     """
        #     Include Unknown (users / items) not shown in train dataset.
        #     If user nor item is not in train dataset --> 0 => Rating will be calculated as 0 (Nan)
        #     One component (item or user) not in the test dataset --> 1 ==> Rating is calculated as the average of P or Q vector.
        #     """
        #     uilist_test = list(zip(test['user'],test['item']))
    
        #     prediction = test.copy()
        #     pred=[]
        #     for val in uilist_test:
        #         if (val[0] not in self.uilist_train[0]) & (val[1]not in self.uilist_train[1]):
        #             P_inner_product = np.full(self.k,0)
        #             Q_inner_product = np.full(self.k,0)
        #             bias_u_product = 0
        #             bias_i_product = 0
        #         elif val[0] not in self.uilist_train[0]:
        #             P_inner_product = np.full(self.k,1)
        #             Q_inner_product = Q_df.loc[val[1]].T
        #             bias_u_product = 0
        #             bias_i_product = int(bi_df.loc[val[1]])
        #         elif val[1] not in self.uilist_train[1]:
        #             P_inner_product = P_df.loc[val[0]]
        #             Q_inner_product = np.full(self.k,1)
        #             bias_u_product = int(bu_df.loc[val[0]])
        #             bias_i_product = 0
        #         else:
        #             P_inner_product = P_df.loc[val[0]]
        #             Q_inner_product = Q_df.loc[val[1]].T
        #             bias_u_product = int(bu_df.loc[val[0]])
        #             bias_i_product = int(bi_df.loc[val[1]])

        #         pred.append((np.dot(P_inner_product,Q_inner_product)+ self.global_mean + bias_u_product + bias_i_product))
        #     prediction['rating'] = pred
        #     test_filtered = test

        return prediction,test_filtered
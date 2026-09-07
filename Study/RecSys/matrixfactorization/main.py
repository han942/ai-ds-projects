from preprocessing import *
from matfac import MatrixFactorization
import numpy as np
from sklearn.metrics import root_mean_squared_error

data_indicator = 0
if data_indicator==0:
    data_list = [['u1.base','u1.test'],
             ['u2.base','u2.test'],
             ['u3.base','u3.test'],
             ['u4.base','u4.test'],
             ['u5.base','u5.test']]

elif data_indicator==1:
    pass

if __name__ == '__main__':
    np.random.seed(42) #재현성

    rmse = []
    prec_at_k = []
    rec_at_k = []
    ndcg_value_k = []
    n = 1
    for b,t in data_list:
        print(f'\nFold {n} / Fold 5 Start')
        train,test,R_train = load_data(b,t,data_indicator=0)

        R_train = R_train.fillna(0)
        k = 10
        lr = 0.0001
        reg_param = 0.02
        epochs = 50

        mf_model  = MatrixFactorization(k,lr,reg_param,epochs)

        print('Start Model Training')
        mf_model.fit(train)
        print('\nModel Training Success')

        #Prediction
        prediction,test = mf_model.predict(test,exclude_unknowns=True)
    
        #Evaluation
        rsme_par = root_mean_squared_error(prediction['rating'].values,test['rating'].values)
        prec_at_k_par = precision_at_k(prediction,test,k=10)
        rec_at_k_par = recall_at_k(prediction,test,k=10)
        ndcg_at_k_par = ndcg_at_k(prediction,test,k=10)

        rmse.append(rsme_par)
        prec_at_k.append(prec_at_k_par)
        rec_at_k.append(rec_at_k_par)
        ndcg_value_k.append(ndcg_at_k_par)
        print(f'RMSE : {rsme_par:.4f} , Precision@k : {prec_at_k_par:.4f} , Recall@k : {rec_at_k_par:.4f} , NDCG@k :{ndcg_at_k_par:.4f}')

        print(f'Fold {n} / Fold 5 Completed')
        n=n+1 #Fold indicator
    print('-'*8,'Model Training Finished','-'*8)
    print(f'RMSE : {np.mean(rmse):.5f} , Precision@k : {np.mean(prec_at_k):.5f} , Recall@k : {np.mean(rec_at_k):.5f},\
           NDCG@k:{np.mean(ndcg_value_k):.5f}')
    
# movielens 100K

#SVD 알아오기

import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path
import numpy as np


def load_data(base,test,data_indicator=0):
    if data_indicator==0:
        path = Path('../../../datafile/Recsys/ml-100k')
        train = pd.read_csv(path / base,sep='\t',
                   usecols=[0,1,2],names=['user','item','rating'],
                   dtype={'user':int,'item':int,'rating':float})
        test = pd.read_csv(path / test,sep='\t',
                   usecols=[0,1,2],names=['user','item','rating'],
                   dtype={'user':int,'item':int,'rating':float})

        R_train = pd.pivot_table(train,values='rating',index='user',columns='item')
    
    elif data_indicator==1:
        path = Path('../../../datafile/Recsys/ml-100k')
        data = pd.read_csv(path / 'u.data',sep='\t',
                   usecols=[0,1,2],names=['user','item','rating'],
                   dtype={'user':int,'item':int,'rating':float})
        
        train = data.groupby('user').sample(frac=0.8,random_state=42)
        train_ind = train.index
        test = data.drop(train_ind)

        print(f'Data Size //  original: {data.shape}, train:{train.shape}, test:{test.shape}')
        R_train = pd.pivot_table(train,values='rating',index='user',columns='item')

    return train,test,R_train
    

def hit_rate_at_k(y_pred,y_true,k): #Predicted item 중에서 Hit 한 item 갯수
    hit = []
    n = len(y_true)
    # k가 -1 또는 데이터 전체보다 크면 k=n으로 보정
    if k > n:
        k = n
    for user_num in y_pred['user'].unique():
        top_test_items = y_true.loc[y_true['user']==user_num].sort_values('rating',ascending=False)[:k]
        top_pred_items = y_pred.loc[y_pred['user']==user_num].sort_values('rating',ascending=False)[:k]
        hit.append(len(set(top_test_items['item']).intersection(set(top_pred_items['item']))) / k)   #Intersection method requires set() data
    return sum(hit) / len(hit)
"""
Sort_values를 method마다 호출하는건 비효율적, 이를 따로 변수로 지정하여 설정하길
"""
def precision_at_k(y_pred,y_true,k):
    prec_at_k = []
    for user_num in y_pred['user'].unique():
        top_test_items = y_true.loc[(y_true['user']==user_num)].sort_values('rating',ascending=False)
        top_test_rel_items = top_test_items
        top_pred_items = y_pred.loc[(y_pred['user']==user_num)].sort_values('rating',ascending=False)[:k]      
        prec_at_k.append(len(set(top_test_rel_items['item']).intersection(set(top_pred_items['item']))) / k)
    return np.mean(prec_at_k)

def recall_at_k(y_pred,y_true,k): #Relevant Item 중에서 Hit (predict)한 item 갯수
    rec_at_k = []
    for user_num in y_pred['user'].unique():
        top_test_items = y_true.loc[(y_true['user']==user_num)].sort_values('rating',ascending=False)
        top_test_rel_items = top_test_items
        top_pred_items = y_pred.loc[(y_pred['user']==user_num)].sort_values('rating',ascending=False)
        top_pred_rel_items = top_pred_items[:k]
        rec_at_k.append(len(set(top_test_rel_items['item']).intersection(set(top_pred_rel_items['item']))) / len(top_test_rel_items) if len(top_test_rel_items) >0 else 0)   #Intersection method requires set() data
    return np.mean(rec_at_k)

def ndcg_at_k(y_pred,y_true,k):
    ndcg_k = []
    for user_num in y_pred['user'].unique():
       top_pred_items = y_pred.loc[(y_pred['user']==user_num)].sort_values('rating',ascending=False)
       pred_sequence = top_pred_items['item'][:k].values

       test_items = y_true.loc[y_true['user']==user_num]
       ideal_rel_score = test_items.sort_values('rating',ascending=False)[:k]['rating'].values
       rel_score = test_items.set_index('item').reindex(pred_sequence)['rating'].values
       dcg_k = np.sum((np.pow(2,rel_score) -1) / np.log2(np.arange(2,len(rel_score)+2)))
       idcg_k = np.sum((np.pow(2,ideal_rel_score) -1) / np.log2(np.arange(2,len(ideal_rel_score)+2)))
       ndcg_k.append(dcg_k / idcg_k if idcg_k>0 else 0)
    
    return np.mean(ndcg_k)

#VAE // mult-VAE
#LightGCN : pyG // recbole
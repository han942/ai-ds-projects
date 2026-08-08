# movielens 100K

#SVD 알아오기

import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path

def load_data(base,test):
    path = Path('../../../datafile/Recsys/ml-100k')
    train = pd.read_csv(path / base,sep='\t',
                   usecols=[0,1,2],names=['user','item','rating'],
                   dtype={'user':int,'item':int,'rating':float})
    test = pd.read_csv(path / test,sep='\t',
                   usecols=[0,1,2],names=['user','item','rating'],
                   dtype={'user':int,'item':int,'rating':float})

    R_train = pd.pivot_table(train,values='rating',index='user',columns='item')
    return train,test,R_train

def precision_at_k(y_pred,y_true,k): #Predicted item 중에서 Hit 한 item 갯수
    hit = []
    for user_num in y_pred['user'].unique():
        top_test_items = y_true.loc[y_true['user']==user_num].sort_values('rating',ascending=False)[:k]
        top_pred_items = y_pred.loc[y_pred['user']==user_num].sort_values('rating',ascending=False)[:k]
        hit.append(len(set(top_test_items['item']).intersection(set(top_pred_items['item']))) / k)   #Intersection method requires set() data
    return sum(hit) / len(hit)

def recall_at_k(y_pred,y_true,k,threshold): #Relevant Item 중에서 Hit (predict)한 item 갯수
    hit = []
    for user_num in y_pred['user'].unique():
        top_test_rel_items = y_true.loc[(y_true['user']==user_num) & (y_true['rating'] >= threshold)].sort_values('rating',ascending=False)[:k]
        top_pred_rel_items = y_pred.loc[(y_pred['user']==user_num) & (y_true['rating'] >= threshold)].sort_values('rating',ascending=False)[:k]
        hit.append(len(set(top_test_rel_items['item']).intersection(set(top_pred_rel_items['item']))) / k)   #Intersection method requires set() data
    return sum(hit) / len(hit)

def ndcg_at_k(y_pred,y_true):
    return 
import pandas as pd
import numpy as np
from config import Config
from collections import Counter
from konlpy.tag import Okt
from tqdm import tqdm
from pathlib import Path

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
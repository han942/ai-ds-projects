from preprocessing import *
from multvae import mult_vae,train_model,evaluate
import numpy as np
from torch.utils.data import DataLoader

if __name__== '__main__':
    np.random.seed(42)
    torch.manual_seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    rmse = []
    prec_at_k = []
    rec_at_k = []
    ndcg_value_k = []

    base=1
    test=1
    n = 1

    print(f'\nFold {n} / Fold 5 Start')
    train,test,R_train,R_test = load_data()
    
    tr_n_users = R_train.shape[0]
    tr_n_items = R_train.shape[1]

    R_train = R_train.values
    R_test = R_test.values

    R_train = MLData(R_train)
    R_test = MLData(R_test)

    train_loader = DataLoader(R_train,batch_size=32,shuffle=True)
    test_loader = DataLoader(R_test,batch_size=32,shuffle=False)

    hidden_dim= [600,200]
    latent_dim = 50
    drop_encoder= 0.1
    drop_decoder= 0.1
    beta=1.0

    mvae_model  = mult_vae(tr_n_users,tr_n_items,hidden_dim,latent_dim,
                           drop_encoder,drop_decoder,beta).to(device)
    
    optimizer = torch.optim.Adam(mvae_model.parameters(),lr=0.001,weight_decay=0.0)
    print('Start Model Training')
    
    train_model(mvae_model,train_loader,optimizer,total_epochs=100,annealing_epochs=10,device=device)
    print('\nModel Training Success')


    top_k_recs = evaluate(mvae_model,test_loader,k=10,device=device)
    print('\n Model Prediction Sucess')


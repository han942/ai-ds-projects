import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np

class mult_vae(nn.Module):
    def __init__(self,n_users,n_items,hidden_dim,latent_dim,drop_encoder,
                 drop_decoder,beta):
        super(mult_vae,self).__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.beta = beta

        self.drop_encoder = drop_encoder
        self.drop_decoder = drop_decoder
       

        #Encoder Layers
        self.enc_input_dropout = nn.Dropout(self.drop_encoder)
        enc_layers = []
        current_enc_dim = self.n_items
        for h_dim in self.hidden_dim:
            enc_layers.append(nn.Linear(current_enc_dim,h_dim))
            enc_layers.append(nn.Tanh())
            current_enc_dim = h_dim
        self.mlp = nn.Sequential(*enc_layers)
        self.enc_fc_mu = nn.Linear(current_enc_dim,self.latent_dim)
        self.enc_fc_log_var = nn.Linear(current_enc_dim,self.latent_dim)

        #Decoder Layers
        dec_hidden_dim = list(reversed(self.hidden_dim))
        dec_layers = []
        current_dec_dim = self.latent_dim
        for h_dim in dec_hidden_dim:
            dec_layers.append(nn.Linear(current_dec_dim,h_dim))
            dec_layers.append(nn.Tanh())
            current_dec_dim = h_dim
        self.dec_mlp = nn.Sequential(*dec_layers)
        self.dec_fc_output = nn.Linear(current_dec_dim,self.n_items)
        self.dec_output_dropout = nn.Dropout(self.drop_decoder)
        pass
    
    def encoder(self,x):
        h = self.enc_input_dropout(x)
        h = self.mlp(h)
        
        return self.enc_fc_mu(h) , self.enc_fc_log_var(h)
    
    def reparametrization(self,mu,log_var):
        """
        For sampling, use Gaussian Dist but needs consider of the prior dist
        """
        std = torch.exp(0.5*log_var)
        eps = torch.randn_like(std)
        z = mu + log_var*eps

        return z
    
    def decoder(self,z):
        h = self.dec_mlp(z)
        h = self.dec_output_dropout(h)

        h = self.dec_fc_output(h)
        return h
    
    def forward(self,train):
        mu,log_var = self.encoder(train)
        if self.training:
            z = self.reparametrization(mu,log_var)
        else:
            z = mu
        logits = self.decoder(z)

        return logits,mu,log_var
    
    def predict(self,x,k=10):
        self.eval()

        with torch.no_grad():
            mu,log_var = self.encoder(x)
            z = mu
            logits = self.decoder(z)
        _,ind = torch.topk(logits,k,dim=1)
        
        return ind
    

    def loss_function(self,logits,x,mu,log_var,beta=1.0):
        log_probs = F.log_softmax(logits,dim=1)
        recon_loss = -torch.sum(log_probs * x, dim=1)
        recon_loss = recon_loss.mean()

        kl_loss = -0.5 * torch.sum(1+log_var-mu.pow(2) - log_var.exp(),dim=1).mean()
        loss = recon_loss + beta*kl_loss

        return recon_loss,kl_loss,loss

def train_model(model,train_loader,optimizer,total_epochs,annealing_epochs,device):
    for epoch in range(total_epochs):
        model.train()
        if epoch <= annealing_epochs:
            beta = epoch * (1.0/annealing_epochs)
        else:
            beta=1.0

        total_loss_epoch=0
        for batch_data in train_loader:
            x = batch_data.to(device)

            optimizer.zero_grad()
            logits,mu,log_var = model(x)

            recon_loss,kl_loss,loss = model.loss_function(logits,x,mu,log_var,beta)
            loss.backward()
            optimizer.step()

            total_loss_epoch += loss.item()
        avg_loss = total_loss_epoch / len(train_loader)
        if (epoch <= annealing_epochs):
            print(f"Epoch {epoch}/{total_epochs} , beta: {beta:.4f}, avg_loss : {avg_loss:.4f}")
        elif (epoch > annealing_epochs) & (epoch %10 ==0):
            print(f"Epoch {epoch}/{total_epochs} , beta: {beta:.4f}, avg_loss : {avg_loss:.4f}")

    print('Training Compelete')

def evaluate(model,test_loader,k,device):
    model.eval()

    all_preds = []
    with torch.no_grad():
        for batch_data in test_loader:
            x = batch_data.to(device)

            topk_ind = model.predict(x,k=k)
            all_preds.append(topk_ind.cpu().numpy())
    
    return np.concatenate(all_preds,axis=0)
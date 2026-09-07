import torch
import torch.nn as nn
import torch.nn.functional as F

class DeepCoNN(nn.Module):
    def __init__(self,config,embedding_matrix):
        super(DeepCoNN,self).__init__()
        self.config = config

        #1.Embedding layer
        vocab_size,embedding_dim = embedding_matrix.shape
        self.embedding = nn.Embedding(vocab_size,embedding_dim)
        self.embedding.weight = nn.Parameter(
            torch.FloatTensor(embedding_matrix)
        )
        self.embedding.weight.requires_grad = True

        #2-1.User Network
        self.user_cnn = nn.Conv1d(
            in_channels=embedding_dim,
            out_channels=config.num_filters,
            kernel_size=config.kernel_size,
            padding=1
        )
        self.user_fc = nn.Linear(config.num_filters,config.latent_dim)

        #2-2. Item Network
        self.item_cnn = nn.Conv1d(
            in_channels=embedding_dim,
            out_channels=config.num_filters,
            kernel_size=config.kernel_size,
            padding=1
        )
        self.item_fc = nn.Linear(config.num_filters,config.latent_dim)

        #Dropout
        self.dropout = nn.Dropout(config.dropout_rate)

        #3.FM Layer (2nd-order latent vectors)
        self.fm_linear = nn.Linear(config.latent_dim*2,1)
        
        self.fm_V = nn.Parameter(
            torch.rand(config.latent_dim*2,config.fm_k)
        )
        self.global_bias = nn.Parameter(torch.zeros(1))

    def forward(self,user_doc,item_doc):
        """
        Args:
            user_doc: (batch_size,max_doc_length)
            item_doc: (batch_size,max_doc_length)

        Returns:
            rating: (batch_size,)
        """

        #User_Net
        user_emb = self.embedding(user_doc) # (B,L,E)
        user_emb = user_emb.transpose(1,2) # (B,E,L) - conv1d input format

        user_conv = F.relu(self.user_cnn(user_emb)) 
        user_pool = F.max_pool1d(user_conv,kernel_size=user_conv.size(2))
        user_pool = user_pool.squeeze(2)

        user_latent = F.relu(self.user_fc(user_pool))
        user_latent = self.dropout(user_latent)

        #item_Net
        item_emb = self.embedding(item_doc) # (B,L,E)
        item_emb = item_emb.transpose(1,2) # (B,E,L) - conv1d input format

        item_conv = F.relu(self.item_cnn(item_emb)) 
        item_pool = F.max_pool1d(item_conv,kernel_size=item_conv.size(2))
        item_pool = item_pool.squeeze(2)

        item_latent = F.relu(self.item_fc(item_pool))
        item_latent = self.dropout(item_latent)

        #concatenate
        z = torch.cat([user_latent,item_latent],dim=1)

        #FM layer
        linear_term = self.fm_linear(z)

        interactions = torch.mm(z,self.fm_V)
        interactions_squared = torch.mm(z**2,self.fm_V**2)

        quadratic_term = 0.5*torch.sum(
            interactions**2-interactions_squared,dim=1,keepdim=True
        )

        #Predict
        rating = self.global_bias + linear_term.squeeze(1) + quadratic_term.squeeze(1)
        
        return rating
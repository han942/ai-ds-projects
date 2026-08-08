class Config:

    max_doc_length = 500
    embedding_dim = 300

    "Model Architecture"
    num_filters = 100
    kernel_size = 3
    latent_dim = 50
    fm_k = 8

    "Traning"

    batch_size = 128
    num_epochs = 50
    learning_rate = 0.002
    dropout_rate = 0.5

    "Other"
    device='cuda'
    random_seed=42
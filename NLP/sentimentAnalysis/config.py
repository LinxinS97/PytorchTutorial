import torch


class Config:
    def __init__(self):
        self.n_vocab = 1002
        self.embed_size = 128
        self.hidden_size = 128
        self.num_layers = 3
        self.drop = 0.8
        self.num_class = 2
        self.pad_size = 32
        self.batch_size = 4096
        self.is_shuffle = True
        self.lr = 0.0056
        self.epochs = 100
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

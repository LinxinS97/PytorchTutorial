import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from model import Model
from dataset import data_loader, TextClass
from config import Config


cfg = Config()

# DATA
data_path = 'data/weibo_senti_100k.csv'
data_stop_path = 'data/hit_stopword.txt'
dict_path = 'data/dict'
dataset = TextClass(data_path, data_stop_path, dict_path)
train_dataloader = data_loader(dataset, cfg)
cfg.pad_size = dataset.max_seq_len

# MODEL
model_text_cls = Model(cfg)
model_text_cls.to(cfg.device)

# LOSS
loss_func = nn.CrossEntropyLoss()
optimizer = optim.Adam(model_text_cls.parameters(), lr=cfg.lr)

# TRAIN
for epoch in range(cfg.epochs):
    for i, batch in enumerate(train_dataloader):
        label, data = batch
        data = torch.tensor(data).to(cfg.device)
        label = torch.tensor(label, dtype=torch.int64).to(cfg.device)

        optimizer.zero_grad()
        pred = model_text_cls.forward(data)
        loss_val = loss_func(pred, label)

        print("epoch:{}, ite:{}, val:{}".format(epoch, i, loss_val))
        loss_val.backward()
        optimizer.step()

    if epoch % 10 == 0:
        torch.save(model_text_cls.state_dict(), 'models/{}.pth'.format(epoch))


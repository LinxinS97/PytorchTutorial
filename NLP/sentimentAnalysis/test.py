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
test_loader = data_loader(dataset, cfg)
cfg.pad_size = dataset.max_seq_len

# MODEL
model_text_cls = Model(cfg)
model_text_cls.to(cfg.device)
model_text_cls.load_state_dict(torch.load('models/10.pth'))
model_text_cls.eval()


# TRAIN
for i, batch in enumerate(test_loader):
    label, data = batch
    data = torch.tensor(data).to(cfg.device)
    label = torch.tensor(label, dtype=torch.int64).to(cfg.device)

    logit = model_text_cls.forward(data)

    pred = torch.argmax(logit, dim=1)
    out = torch.eq(pred, label)
    print('acc:{}'.format(out.sum() * 1.0 / pred.size()[0]))




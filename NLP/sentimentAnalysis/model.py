import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np


class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        self.embedding = nn.Embedding(config.n_vocab,  # 字典大小
                                      config.embed_size,  # embedding维度
                                      padding_idx=config.n_vocab - 1)  # padding之后的词的索引
        self.lstm = nn.LSTM(config.embed_size,  # embedding维度
                            config.hidden_size,  # 隐藏层数量
                            config.num_layers,   # 层数
                            bidirectional=True,  # 是否双向LSTM
                            batch_first=True,
                            dropout=config.drop)

        self.maxpool = nn.MaxPool1d(config.pad_size)
        self.fc = nn.Linear(config.hidden_size * 2 + config.embed_size,  # 双向lstm + embedding size
                            config.num_class)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        embed = self.embedding(x)  # output: [batchsize, seq_len, embed_size]
        out, _ = self.lstm(embed)
        out = torch.cat((embed, out), 2)
        out = F.relu(out)
        out = out.permute(0, 2, 1)  # 交换维度
        out = self.maxpool(out).reshape(out.size()[0], -1)
        out = self.fc(out)
        out = self.softmax(out)
        return out


if __name__ == '__main__':
    from config import Config
    cfg = Config()
    cfg.pad_size = 640
    model_test = Model(config=cfg)
    input_tensor = torch.tensor([i for i in range(640)]).reshape([1, 640])
    out_tensor = model_test.forward(input_tensor)
    print(out_tensor.size())
    print(out_tensor)

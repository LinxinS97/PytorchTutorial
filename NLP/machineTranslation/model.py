import torch
import torch.nn as nn
import torch.nn.functional as F
from dataset import MAX_LEN

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)

    def forward(self, x, hidden):
        embedding = self.embedding(x).view(1, 1, -1)
        out = embedding
        out, hidden = self.gru(out, hidden)
        return out, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)


class Decoder(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(Decoder, self).__init__()
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)
        self.linear = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, x, hidden):
        out = self.embedding(x).view(1, 1, -1)
        out = F.relu(out)
        out, hidden = self.gru(out, hidden)
        out = self.softmax(self.linear(out[0]))
        return out, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)


class AttentionDecoder(nn.Module):
    def __init__(self, hidden_size, output_size, dropout=0.1, max_len=MAX_LEN):
        super(AttentionDecoder, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout
        self.max_len = max_len

        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.attn = nn.Linear(self.hidden_size * 2, self.max_len)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)

        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size)
        self.linear = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, x, hidden, encoder_outputs):
        embedding = self.embedding(x).view(1, 1, -1)
        embedding = self.dropout(embedding)

        attn_weight = F.softmax(
            self.attn(torch.cat([embedding[0], hidden[0]], 1)),
            dim=1
        )
        attn_applied = torch.bmm(
            attn_weight.unsqueeze(0),
            encoder_outputs.unsqueeze(0)
        )
        out = torch.cat([embedding[0], attn_applied[0]], dim=1)
        out = self.attn_combine(out).unsqueeze(0)
        out = F.relu(out)
        out, hidden = self.gru(out, hidden)
        out = F.log_softmax(self.linear(out[0]), dim=1)

        return out, hidden, attn_weight

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)


if __name__ == '__main__':
    encoder = Encoder(5000, 256)
    decoder = Decoder(256, 5000)
    attn_decoder = AttentionDecoder(256, 5000)

    tensor_in = torch.tensor([12, 14, 16, 18], dtype=torch.long).view(-1, 1)
    hidden_in = torch.zeros(1, 1, 256)
    c_in = torch.zeros(1, 1, 256)
    encoder_out, encoder_hidden = encoder(tensor_in[0], hidden_in, c_in)

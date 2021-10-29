import random
import time
import torch
import torch.nn as nn
from torch import optim
from dataset import readLangs, SOS_TOKEN, EOS_TOKEN
from model import Encoder, AttentionDecoder, device
from utils import timeSince

MAX_LEN = 10 + 1  # 插入一个结尾符所以+1
path = 'data/cmn.txt'
lang1 = 'en'
lang2 = 'cn'

in_lang, out_lang, pairs = readLangs(lang1, lang2, path)


def list2tensor(input_lang, data):
    index_in = [input_lang.word2index[word] for word in data.split(' ')]
    index_in.append(EOS_TOKEN)  # 加入一个终止符
    tensor_in = torch.tensor(index_in, dtype=torch.long, device=device).view(-1, 1)  # n*1
    return tensor_in


def pair2tensor(pair):
    in_tensor = list2tensor(in_lang, pair[0])
    out_tensor = list2tensor(out_lang, pair[1])
    return in_tensor, out_tensor


# MODEL
hidden_size = 256
encoder = Encoder(in_lang.n_words, hidden_size).to(device)
decoder = AttentionDecoder(hidden_size,
                           out_lang.n_words,
                           max_len=MAX_LEN,
                           dropout=0.1).to(device)

encoder.load_state_dict(torch.load('models/encoder_200000.pth'))
decoder.load_state_dict(torch.load('models/decoder_200000.pth'))

# TRAIN
n_iters = 10
train_sen_pairs = [random.choice(pairs) for i in range(n_iters)]
training_pairs = [pair2tensor(train_sen_pairs[i]) for i in range(n_iters)]  # 随机生成1000000个样本对

for iter in range(n_iters):
    input_tensor, _ = training_pairs[iter]

    # ENCODER
    encoder_hidden = (encoder.initHidden(), encoder.initC())
    input_len = input_tensor.size(0)
    encoder_outputs = torch.zeros(MAX_LEN, encoder.hidden_size, device=device)

    # 这是一个标准的rnn的推理流程
    # 每次从输入序列中提取一个输入扔进rnn，然后从rnn中获取当前的输出和对应的hidden，并且将hidden用于下一次的输入
    for ei in range(input_len):
        encoder_output, encoder_hidden = encoder(input_tensor[ei], encoder_hidden[0], encoder_hidden[1])
        encoder_outputs[ei] = encoder_outputs[0, 0]

    # DECODER
    decoder_input = torch.tensor([[SOS_TOKEN]], device=device)  # decoder的输入为起始符
    decoder_hidden = encoder_hidden

    decoder_words = []
    for di in range(MAX_LEN):
        decoder_output, decoder_hidden, decoder_attn = decoder(
            decoder_input, decoder_hidden[0], decoder_hidden[1], encoder_outputs
        )
        topV, topi = decoder_output.topk(1)
        decoder_input = topi.squeeze().detach()

        if decoder_input.item() == EOS_TOKEN:
            decoder_words.append("<EOS>")
            break
        else:
            decoder_words.append(out_lang.index2word[topi.item()])

    print(train_sen_pairs[iter][0])
    print(train_sen_pairs[iter][1])
    print(decoder_words)


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


# LOSS
def loss_func(input_tensor, output_tensor, encoder, decoder,
              encoder_optimizer,
              decoder_optimizer,
              criterion):
    # ENCODER
    encoder_hidden = encoder.initHidden()
    encoder_optimizer.zero_grad()
    input_len = input_tensor.size(0)
    output_len = output_tensor.size(0)
    encoder_outputs = torch.zeros(MAX_LEN, encoder.hidden_size, device=device)

    # 这是一个标准的rnn的推理流程
    # 每次从输入序列中提取一个输入扔进rnn，然后从rnn中获取当前的输出和对应的hidden，并且将hidden用于下一次的输入
    for ei in range(input_len):
        encoder_output, encoder_hidden = encoder(input_tensor[ei], encoder_hidden)
        encoder_outputs[ei] = encoder_outputs[0, 0]

    # DECODER
    decoder_input = torch.tensor([[SOS_TOKEN]], device=device)  # decoder的输入为起始符
    decoder_hidden = encoder_hidden
    decoder_optimizer.zero_grad()

    # 这里加入了一个teacher因子
    # RNN网络在刚开始训练的时候标签非常不准确，因此，为了让网络能够更快的收敛
    # 这里使用一个随机数来决定decoder使用的是真实标签还是decoder计算的结果
    use_teacher_forcing = True if random.random() < 0.5 else False

    loss = 0
    if use_teacher_forcing:  # 使用真实标签作为输入
        for di in range(output_len):
            decoder_output, decoder_hidden, decoder_attn = decoder(
                decoder_input, decoder_hidden, encoder_outputs
            )
            loss += criterion(decoder_output, output_tensor[di])
            decoder_input = output_tensor[di]
    else:  # 用decoder的预测结果作为输入
        for di in range(output_len):
            decoder_output, decoder_hidden, decoder_attn = decoder(
                decoder_input, decoder_hidden, encoder_outputs
            )
            loss += criterion(decoder_output, output_tensor[di])
            topV, topi = decoder_output.topk(1)
            decoder_input = topi.squeeze().detach()

            if decoder_input.item() == EOS_TOKEN:
                break

    loss.backward()
    encoder_optimizer.step()
    decoder_optimizer.step()
    return loss.item() / output_len


# MODEL
hidden_size = 256
lr = 0.01
encoder = Encoder(in_lang.n_words, hidden_size).to(device)
decoder = AttentionDecoder(hidden_size,
                           out_lang.n_words,
                           max_len=MAX_LEN,
                           dropout=0.1).to(device)

encoder_optimizer = optim.SGD(encoder.parameters(), lr=lr)
decoder_optimizer = optim.SGD(decoder.parameters(), lr=lr)

scheduler_encoder = torch.optim.lr_scheduler.StepLR(encoder_optimizer, step_size=1, gamma=0.95)
scheduler_decoder = torch.optim.lr_scheduler.StepLR(decoder_optimizer, step_size=1, gamma=0.95)

criterion = nn.NLLLoss()

# TRAIN
n_iters = 1000000
training_pairs = [pair2tensor(random.choice(pairs)) for _ in range(n_iters)]  # 随机生成1000000个样本对

print_every = 100  # 每循环1000次打印一次log
save_every = 10000
print_loss_total = 0
start = time.time()

for iter in range(1, n_iters + 1):
    pair = training_pairs[iter - 1]
    input_tensor = pair[0]
    output_tensor = pair[1]

    loss = loss_func(input_tensor,
                     output_tensor,
                     encoder,
                     decoder,
                     encoder_optimizer,
                     decoder_optimizer,
                     criterion)
    print_loss_total += loss
    if iter % print_every == 0:
        print_loss_avg = print_loss_total / print_every
        print_loss_total = 0
        print('iter:{}, loss:{}, {}%, {}'.format(iter,
                                                 print_loss_avg,
                                                 iter / n_iters * 100,
                                                 timeSince(start, iter / n_iters)))
    # SAVE
    if iter % save_every == 0:
        torch.save(encoder.state_dict(), 'models/encoder_{}.pth'.format(iter))
        torch.save(decoder.state_dict(), 'models/decoder_{}.pth'.format(iter))

    # 调整学习率
    if iter % 10000:
        scheduler_decoder.step()
        scheduler_decoder.step()

from abc import ABC

from torch.utils.data import Dataset, DataLoader
import jieba
import numpy as np


def read_dict(voc_dict_path):
    voc_dict = {}
    dict_list = open(voc_dict_path, encoding='utf-8').readlines()
    for item in dict_list:
        item = item.split(',')
        # {word: id}
        voc_dict[item[0]] = int(item[1].strip())

    return voc_dict


def load_data(data_path, data_stop_path):
    data_list = open(data_path, encoding='utf-8').readlines()[1:]
    stops_word = open(data_stop_path, encoding='utf-8').readlines()

    # 将停用词存入list
    stops_word = [line.strip() for line in stops_word]
    stops_word.append(' ')
    stops_word.append('\n')

    voc_dict = dict()
    data = []
    max_seq_len = 0
    for item in data_list:
        # dataset中第0个是标签，第2个开始是文本
        label = item[0]
        content = item[2:].strip()
        # jieba分词
        seg_list = jieba.cut(content, cut_all=False)

        seg_res = []
        for seg_item in seg_list:
            # 忽略停用词
            if seg_item in stops_word:
                continue
            seg_res.append(seg_item)

            # 词频统计
            if seg_item in voc_dict.keys():
                voc_dict[seg_item] = voc_dict[seg_item] + 1
            else:
                voc_dict[seg_item] = 1
        if len(seg_res) > max_seq_len:
            max_seq_len = len(seg_res)
        data.append([label, seg_res])
    return data, max_seq_len


class TextClass(Dataset):
    def __init__(self, data_path, data_stop_path, voc_dict_path):
        self.data_path = data_path
        self.data_stop_path = data_stop_path
        self.voc_dict = read_dict(voc_dict_path)
        self.data, self.max_seq_len = load_data(self.data_path, self.data_stop_path)

        np.random.shuffle(self.data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = self.data[idx]
        label = int(data[0])
        word_list = data[1]
        input_idx = []

        for word in word_list:
            if word in self.voc_dict.keys():
                input_idx.append(self.voc_dict[word])
            else:
                input_idx.append(self.voc_dict['<UNK>'])
        if len(input_idx) < self.max_seq_len:
            input_idx += [self.voc_dict['<PAD>'] for _ in range(self.max_seq_len - len(input_idx))]

        data = np.array(input_idx)
        return label, data


def data_loader(dataset, config):
    # data_path = 'data/weibo_senti_100k.csv'
    # data_stop_path = 'data/hit_stopword.txt'
    # dict_path = 'data/dict'
    return DataLoader(dataset, batch_size=config.batch_size, shuffle=config.is_shuffle)


if __name__ == "__main__":
    data_path = 'data/weibo_senti_100k.csv'
    data_stop_path = 'data/hit_stopword.txt'
    dict_path = 'data/dict'

    train_dataloader = data_loader(data_path, data_stop_path, dict_path)
    for i, batch in enumerate(train_dataloader):
        print(batch)
        break


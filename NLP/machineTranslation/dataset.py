import jieba
from utils import normalizeString, cht_to_chs

SOS_TOKEN = 0  # 起始符
EOS_TOKEN = 1  # 终止符
MAX_LEN = 10  # 字符串最大长度


# 对字典的统计
class Lang:
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {
            SOS_TOKEN: "SOS",  # 起始符
            EOS_TOKEN: "EOS"  # 终止符
        }
        self.n_words = 2  # 单词索引，向后延续，每增加一个单词+1

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1

    def addSentence(self, sentence):
        for word in sentence.split(' '):
            self.addWord(word)


def readLangs(lang1, lang2, path):
    lines = open(path, encoding='utf-8').readlines()

    lang1_cls = Lang(lang1)
    lang2_cls = Lang(lang2)

    pairs = []
    for line in lines:
        line = line.split('\t')
        sentence1 = normalizeString(line[0])
        sentence2 = cht_to_chs(line[1])
        seg_list = jieba.cut(sentence2, cut_all=False)
        sentence2 = ' '.join(seg_list)

        if len(sentence1.split(' ')) > MAX_LEN:
            continue

        if len(sentence2.split(' ')) > MAX_LEN:
            continue

        pairs.append([sentence1, sentence2])
        lang1_cls.addSentence(sentence1)
        lang2_cls.addSentence(sentence2)

    return lang1_cls, lang2_cls, pairs


if __name__ == '__main__':
    lang1_cls, lang2_cls, pairs = readLangs('en', 'cn', 'data/cmn.txt')
    print(len(pairs))
    print(lang1_cls.n_words)
    # print(lang1_cls.index2word)

    print(lang2_cls.n_words)
    # print(lang2_cls.index2word)

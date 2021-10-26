# dataset: https://github.com/SophonPlus/ChineseNlpCorpus/blob/master/datasets/weibo_senti_100k/intro.ipynb

import jieba

data_list = open("weibo_senti_100k.csv", encoding='utf-8').readlines()[1:]
stops_word = open('hit_stopword.txt', encoding='utf-8').readlines()

# 将停用词存入list
stops_word = [line.strip() for line in stops_word]
stops_word.append(' ')
stops_word.append('\n')

min_seq = 1
top_n = 1000
UNK = "<UNK>"
PAD = "<PAD>"

voc_dict = dict()
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
    # print(content)
    # print(seg_res)

# 定义字典，选出频度大于min_seq的单词，并对其进行排序
voc_list = sorted([_ for _ in voc_dict.items() if _[1] > min_seq], key=lambda x: x[1], reverse=True)[:top_n]
voc_dict = {word_count[0]: idx for idx, word_count in enumerate(voc_list)}
voc_dict.update({UNK: len(voc_dict), PAD: len(voc_dict) + 1})

print(voc_dict)

# 将词频字典写入文件
ff = open('dict', 'w', encoding='utf-8')
for item in voc_dict.keys():
    ff.writelines("{},{}\n".format(item, voc_dict[item]))
ff.close()

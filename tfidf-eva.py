# -*- coding: utf-8 -*-
# 将输入的语料中的所有词的tf-idf权重输出，过滤候选词
import csv
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import evaluate,pretreatText

def tfidf():
    # 读取文件：
    fdir = '/home/ubuntu/文档/corpus/'
    text_csv = 'tokenizedContent-utf8.csv'
    # 1.文本(已分词，用";"隔开)
    sentences = ''  # 保存每一行的文本内容
    tokenized_sentences = []
    with open(fdir + text_csv, encoding='utf-8') as f:
        data = csv.reader(f)
        for content in data:
            sentences = content[0].replace(';', ' ')
            tokenized_sentences.append([sentences])
            # print('sentences:', sentences)
            sentences = ''

    # print('tokenized_sentences:',tokenized_sentences[:2])
    tfidf_dict = []
    temp = []
    for i in np.arange(len(tokenized_sentences)):
        words_weight = []
        word_weight = []
        # 该类会将文本中的词语转换为词频矩阵，矩阵元素a[i][j] 表示j词在i类文本下的词频
        vectorizer = CountVectorizer()
        # 该类会统计每个词语的tf-idf权值
        transformer = TfidfTransformer()
        tfidf = transformer.fit_transform(vectorizer.fit_transform(tokenized_sentences[i]))
        # print('tfidf:',tfidf)
        # 获取词袋模型中的所有词语
        word = vectorizer.get_feature_names()
        # print('word:', word)
        # 将tf-idf矩阵抽取出来，元素a[i][j]表示j词在i类文本中的tf-idf权重
        weight = tfidf.toarray()
        # 打印每类文本的tf-idf词语权重
        for i in range(len(weight)):
            # print(u"-------这里输出词语tf-idf权重------")
            for j in range(len(word)):
                word_weight = [word[j], weight[i][j]]
                words_weight.append(word_weight)
                # print(word[j], weight[i][j])
        # print('words_weight:',words_weight)
        # 权重排序，找出最大的n个
        topK = 5
        result = sorted(words_weight, key=lambda w: w[1], reverse=True)[:topK]
        # print('result:',result)
        for list in result:
            temp.append(list[0])

        tfidf_dict.append(temp)
        temp = []

    print('tfidf_dict:', tfidf_dict, ' len:', len(tfidf_dict))
    return tfidf_dict

# test
result = tfidf()

train_x, train_y, vocabulary_size, index_to_word, tokenized_sentences,keywords, word_flag = pretreatText.dataset()
sample_train_idx = len(train_x) // 10 * 7
sample_eva_idx = sample_train_idx-1  # 30% evaluate
print('sample_train_num:', sample_train_idx, ' sample_eva_num:', sample_eva_idx)
# evaluate precision & recall
evaluate.evaluate_rnn(keywords[sample_eva_idx:-1],result[sample_eva_idx:-1])
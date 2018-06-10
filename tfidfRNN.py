# -*- coding: utf-8 -*-
# 将输入的语料中的所有词的tf-idf权重输出，过滤候选词
import csv
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np

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

        # 将词和它的权重构成为dict
        tfidf_temp = dict((w, we) for w, we in (w_w for w_w in words_weight))
        tfidf_dict.append(tfidf_temp)
    # print('tfidf_dict:',tfidf_dict[:1],' len:',len(tfidf_dict[:1]))
    return tfidf_dict

# test
# tfidf()
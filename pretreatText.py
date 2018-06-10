# -*- coding: utf-8 -*-
# 预处理文本
# 修改记录：
# 1.增加word2vec的模型加载，并应用到数据集生成中
import numpy as np
import csv
import loadW2vModel
import jieba.analyse
import tokenizeText

# 导入自定义词典
jieba.load_userdict('/home/ubuntu/文档/dicts/dict.txt')
# 启用停止词词典
stop_words_path = "/home/ubuntu/文档/dicts/extra_dict/stop_words.txt"
jieba.analyse.set_stop_words(stop_words_path)
stop_words = [line.strip() for line in open(stop_words_path, encoding='utf-8').readlines()]
# 自定义语料库
jieba.analyse.set_idf_path("/home/ubuntu/文档/dicts/extra_dict/idf.txt.big")

def dataset():
    # 加载word2vec模型
    model = loadW2vModel.load_w2v_model()

    # 读取文件：
    fdir = '/home/ubuntu/文档/corpus/'
    text_csv = 'tokenizedContent-utf8.csv'
    flag_csv = 'tokenizedContent&POS-utf8.csv'  # POS excel
    key_csv = 'keywords-utf8.csv'

    # 1.文本(已分词，用";"隔开)
    sentences = []  # 保存每一行的文本内容
    tokenized_sentences = []  # 保存分词好的每一行的文本内容
    with open(fdir+text_csv, encoding='utf-8') as f:
        data = csv.reader(f)
        for content in data:
            sentences.append(content[0].split(';'))
            # print('content:',content)
    tokenized_sentences = sentences
    # print('sentences:',sentences)

    # 2.keyword
    keywords = []
    with open(fdir+key_csv, encoding='utf-8') as f:
        data = csv.reader(f)
        for content in data:
            # print('content:',content[0])
            keywords.append(content[0].split(';'))
    # print('keywords:',keywords)

    # 3.POS
    flags = []
    with open(fdir+flag_csv, encoding='utf-8') as f:
        data = csv.reader(f)
        for content in data:
            # print('content:',content[0])
            flags.append(content[0].split(';'))
    # print('flags:',flags)

    index_to_word = []  # 已去停用词，未去重复词
    for sent in tokenized_sentences:
        for word in sent:
            # if word not in stop_words:  # 去停用词
                index_to_word.append(word)
                # print('word:',word)

    index_to_word = sorted(set(index_to_word),key=str.lower)  # 去重复词，并排序
    # print('index_to_word:',index_to_word)
    index_to_word.append("UNKNOWN_TOKEN")
    vocabulary_size = model.vector_size
    print('vocabulary size:', vocabulary_size)

    # word_to_index = dict([(w, i) for i, w in enumerate(index_to_word)])  # 转为word:index的形式

    # 把所有词表外的词都标记为unknown_token
    for sent in tokenized_sentences:
        for i, word in enumerate(sent):
            sent[i] = [word if word in index_to_word else "UNKNOWN_TOKEN"]

    # print('tokenized_sentences with unknown:',tokenized_sentences[0])
    # print('tokenized_sentences with unknown len:',len(tokenized_sentences[1]))
    # print('flags:', flags[:2])
    # print('flags len:',len(flags[1]))

    # 生成训练数据-标签
    Y_train = []
    for sentence in tokenized_sentences:
        temp_Y = np.zeros(len(sentence))
        for keys in keywords:
            # print('keys:',keys)
            for key in keys:
                # print('key:',key)
                for sent in sentence:
                    if (key in sent):
                        temp_Y[sentence.index(sent)] = 1#/len(keys)
                        # print('sentence.index(sent):',sentence.index(sent))
        Y_train.append(temp_Y)
        temp_Y = []

    # 将数据转为word2vec的词向量
    temp_x = []
    train_x = []
    # flags_new = []
    # temp_flag = []
    for i in np.arange(len(tokenized_sentences)):
        # print('tokenized_sentences:',tokenized_sentences)
        for word in tokenized_sentences[i]:
            # print('model word len:',len(model[word[0]]))
            if(hasattr(model,word[0]) == True):
                temp_x.append(model[word[0]])
                # print('shape:',model[word[0]].shape)
            else:
                temp_x.append(np.random.uniform(-0.5/model.vector_size,0.5/model.vector_size,model.vector_size))
            # print('tokenized_sentences[i].index(word):', tokenized_sentences[i].index(word))
            # temp_flag.append(flags[i][tokenized_sentences[i].index(word)])  # 词性对应存入词性的list
        train_x.append(temp_x)
        # flags_new.append(temp_flag)
        # print('flags_new:', flags_new)
        temp_x = []
        # temp_flag = []

    # test len
    # for i in np.arange(len(train_x)):
    #     if(len(train_x[i])==len(tokenized_sentences[i])):
    #         print('train x equal:',len(train_x[i]))
    #     else:
    #         print('not equal,i:',i,' len(train_x[i])',len(train_x[i]),
    #               ' len(tokenized_sentences[i])',len(tokenized_sentences[i]))

    # print('Y:',Y_train[0][1],' len:',len(Y_train[0]))
    # print('train_x:', train_x[279],' len:',len(train_x[279]))
    return train_x,Y_train,vocabulary_size,index_to_word,tokenized_sentences,keywords,flags

# dataset()
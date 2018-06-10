# -*- coding: utf-8 -*-
# 加载训练好的word2vec模型
import warnings
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')

import logging, os.path, sys, csv, gensim.models
from gensim.models import Word2Vec, word2vec, KeyedVectors


def load_w2v_model():
    fdir = "/home/ubuntu/文档/models/"
    # outp1为训练好的模型
    # outp1 = 'content-w2v.model'
    # outp2 = 'content-w2v.vector'
    # outp3 = 'content-w2v.bin'
    outp1 = 'news_sohu.model'
    # outp2 = 'keywords-w2v.model'

    # 加载模型
    model = word2vec.Word2Vec.load(fdir + outp1)
    # model2 = word2vec.Word2Vec.load(fdir + outp2)
    # print('similarity:', model.similarity(u'特朗普', u'希拉里'))
    print('dim:', model.vector_size)
    # print('vector:', model['希拉里'])
    # print('shape:',model['希拉里'].shape)

    return model

load_w2v_model()
# -*- coding: utf-8 -*-
# 创建于2018.3.18 16:16
# 用于对RNN结果进行评测：召回率、准确率
import pretreatText
import numpy as np

# train_x, Y_train, vocabulary_size, index_to_word, tokenized_sentences = pretreatText.dataset()

def evaluate_rnn(y_true,y_predict):
    '''
    计算召回率、准确率
    :param y_true: 正确结果
    :param y_predict: 预测结果
    :return :
    '''
    correct_num = 0  # 通过算法提取到的正确的关键词总数量
    predict_num = sum(len(words) for words in y_predict)  # 通过算法提取到的关键词总数量
    true_num = sum(len(words) for words in y_true)  # 人工标注的完全的关键词数量
    print('predict_num:',predict_num,' true_num:',true_num)
    # print('y_predict:',y_predict)
    for i in np.arange(len(y_true)):
        for word in y_predict[i]:
            # print('word:',word)
            # print('y_true[i]:',y_true[i])
            if(word[0] in y_true[i]):
                correct_num += 1

    # 计算召回率
    recall = correct_num / true_num
    # 计算准确率
    precision = correct_num / predict_num
    print('recall:',recall,' precision:',precision)
    # return recall,precision

# evaluate_rnn([[['a'],['b']],[['c'],['d']]],[[['a'],['b'],['dd']],[['c'],['d']]])
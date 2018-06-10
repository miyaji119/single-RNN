# -*- coding: utf-8 -*-
# 创建于2018.3.26 23:56
# 1.封装数据集的生成代码为dataset()
# 2.加入word2vec
import numpy as np
import rnnUtils, pretreatText, tfidfRNN, saveLossLog, evaluate
import sys, datetime, operator
from rnnUtils import save_model_parameters, load_model_parameters
import math

P_RATE = 0.75  # predict rate 75%
T_RATE = 8  # tf-idf rate 25%
class RNNNumpy(object):
    def __init__(self,word_dim,hidden_dim=100):#,bptt_truncate=4):
        """
        初始化RNN类
        权重矩阵的维度：
        U -- (n_h,n_x)
        V -- (1,n_h)
        W -- (n_h,n_h)
        :param word_dim: 输入词向量的维度，即n_x
        :param hidden_dim: 隐藏层的节点数，即n_h
        """
        self.word_dim = word_dim
        self.hidden_dim = hidden_dim
        # self.bptt_truncate = bptt_truncate
        # 初始化U，V，W
        self.U = np.random.uniform(-np.sqrt(1. / word_dim), np.sqrt(1. / word_dim), (hidden_dim, word_dim))
        # self.V = np.random.uniform(-np.sqrt(1. / hidden_dim), np.sqrt(1. / hidden_dim), (word_dim, hidden_dim))
        self.V = np.random.uniform(-np.sqrt(1. / hidden_dim), np.sqrt(1. / hidden_dim), (1, hidden_dim))
        self.W = np.random.uniform(-np.sqrt(1. / hidden_dim), np.sqrt(1. / hidden_dim), (hidden_dim, hidden_dim))
    # end of __init__

    def forward_propagation(self,X):
        """
        前向传播
        :param self:
        :param X: 输入文本对应的词向量的list，其长度为文本中word的数量，每个word的词向量作为一个时刻t的输入
        :return o: 每一时刻的网络输出
        :return s: 每一时刻的隐藏层节点的输出值，(t,n_h)，s[t][i]表示t时刻隐藏层第i个节点的输出值
        """
        T = len(X)  # 时间步的总数量，即一个文本中word的数量
        # 在前向传播中，将隐藏层的状态保存在一个变量s中
        s = np.zeros((T+1,self.hidden_dim))  # 初始化s为全0的矩阵
        s[-1] = np.zeros((1,self.hidden_dim))

        # o = np.zeros((T,self.word_dim))  # 每个时刻的输出
        o = np.zeros(T)
        # 时刻循环
        for t in np.arange(T):
            temp_x = np.dot(self.U,X[t])
            # print('temp_x shape:', temp_x.shape)
            # print('s[t-1]:', s[t - 1], 'shape:', s[t - 1].shape)
            temp_s =  np.dot(self.W,s[t-1])
            # print('temp_s shape:', temp_s.shape)
            temp = temp_x + temp_s
            # print('temp shape:', temp.shape)
            s[t] = np.tanh(temp)
            # t时刻的输出
            # o[t] = rnnUtils.softmax(np.dot(self.V,s[t]))
            o[t] = np.dot(self.V, s[t])
        # print('o:', o)
        softmax_o = rnnUtils.softmax(o)  # 将包含每个时刻输出概率到list o进行softmax计算
        # print('softmax_o.shape:', softmax_o.shape,' softmax_o:',softmax_o)
        return [softmax_o,s]
    # end of forward_propagation

    def cal_total_loss(self,X,Y):
        """
        计算一个样本的损失和
        :param self:
        :param X: 单个样本的输入
        :param Y: 单个样本的输出
        :return L: 损失和
        """
        L = 0  # 损失值
        o,s = self.forward_propagation(X)
        # p_words = o[:,-1]
        # L += -1 * np.sum(np.log(p_words)*Y)
        L += -1 * np.sum(np.log(o) * Y)
        # print('cal_total_loss L:',L)
        return L
    # end of cal_total_loss

    def cal_loss(self,X,Y):
        """
        计算所有样本的损失和
        :param self:
        :param X: 所有样本的输入
        :param Y: 所有样本的输出
        :return: 所有样本的loss
        """
        N = np.sum((len(Y[i]) for i in np.arange(len(Y))))
        # N = np.sum(len(Y))
        loss = 0
        for i in np.arange(len(Y)):
            loss += self.cal_total_loss(X[i],Y[i])
        loss = loss / N
        # print('cal_loss loss:',loss)
        return loss
    # end of cal_loss

    def bptt(self,X,Y,bptt_truncate=8):
        """
        BPTT，通过时间反向传播算法
        :param self:
        :param X: 单个样本的输入
        :param Y: 单个样本的输入
        :return dU：损失函数对U矩阵的导数，维度与U相同
        :return dV：损失函数对V矩阵的导数，维度与V相同
        :return dW：损失函数对W矩阵的导数，维度与W相同
        """
        T = len(Y)
        o,s = self.forward_propagation(X)
        # 初始化梯度
        dU = np.zeros(self.U.shape)
        dV = np.zeros(self.V.shape)
        dW = np.zeros(self.W.shape)
        delta_o = o
        delta_o[np.arange(T)] -= 1   # delta_o[t] = o[t] - 1
        # 反向传播
        for t in np.arange(T)[::-1]:
            dV += np.outer(delta_o[t], s[t].T)  # dV += (delta_o[t])xs[t].T
            # 初始化导数计算: dL/d(Ux[T]+Ws[T-1])，时刻T的导数
            # delta_t = dL/d(Ux[T]+Ws[T-1]) = dL/d(s[T]) = tanh'(s[T])xV.Tx(delta_o[T])
            delta_t = np.dot(self.V.T,[delta_o[t]]) * (1 - s[t]**2)
            for bptt_step in np.arange(max(0,t-bptt_truncate),t+1)[::-1]:
                # dW += diag(1-s[t]**2)x(dL/d(s[t]))xs[t-1].T
                dW += np.outer(delta_t,s[bptt_step-1].T)
                # dU += diag(1-s[t]**2)x(dL/d(s[t]))xX[t].T
                dU += np.outer(delta_t,X[bptt_step].T)
                # 更新前一时刻的dL/d(Ux[t]+Ws[t-1])
                delta_t = np.dot(self.W.T,delta_t)*(1-s[bptt_step-1]**2)
        return [dU,dV,dW]
    # end of bptt

    def predict(self,X):
        """
        prototype-进行预测，取概率符合的词index
        :param self:
        :param X: 多个样本的输入
        :return result: 符合条件的word的index
        """
        result = []
        predict = []  # 保存一个样本的预测结果
        for i in np.arange(len(X)):
            o_i,s_i = self.forward_propagation(X[i])
            # 取一个样本中所有word输入完成后的概率结果,idx是word在词表中的index
            for idx,v in enumerate(o_i):
                # print('idx-v:', idx, '-', v)
                if(v>=SCOPE):
                    print('idx-v:', idx, '-', v)
                    predict.append(idx)
            # print('x_i predict:',i,'-',predict)
            result.append(predict)
            predict = []  # 清空
        return result
    # end of predict

    def predict_with_tfidf(self,X):
        """
        进行预测，取概率前20%的词index
        :param self:
        :param X: 多个样本的输入
        :return result: 符合条件的word的index
        """
        result = []
        predict = []  # 保存一个样本的预测结果
        for i in np.arange(len(X)):
            o_i,s_i = self.forward_propagation(X[i])
            topK = math.ceil(len(o_i) * P_RATE)  # 词数量范围
            # 将概率结果保存为(index,value)的形式
            word_dict = [(idx,value) for idx,value in enumerate(o_i)]
            # 取概率前10%
            predict = sorted(word_dict, key=lambda w: w[1], reverse=True)[:topK]
            # print('temp_o:',predict)
            result.append(predict)
            # predict = []  # 清空
        return result
    # end of predict

    def gradient_check(self,X,Y,h=0.0001,error_threshold=0.01):
        """
        梯度检查
        :param X:
        :param Y:
        :param h: 步长
        :param error_threshold: 误差阈值
        """
        # 用反向传播计算梯度
        bptt_gradients = self.bptt(X,Y)
        # 要检查的参数
        model_parameters = ['U','V','W']
        # 检查每个参数
        for pIdx,pName in enumerate(model_parameters):
            # 获取属性对应的值
            parameter = operator.attrgetter(pName)(self)
            print('performing gradient check for parameter %s with size %d.' % (pName,np.prod(parameter.shape)))
            # 迭代parameter矩阵中的每个元素，如(0,0)，(0,1)，···，对parameter进行多重索引，且可读可写
            it = np.nditer(parameter,flags=['multi_index'],op_flags=['readwrite'])
            while not it.finished:
                ix = it.multi_index  # 表示输出元素的索引
                # 保存原来的值
                original_value = parameter[ix]
                # 估计梯度计算，(f(x+h)-f(x-h))/(2*h)
                parameter[ix] = original_value + h
                # gradplus = self.cal_total_loss(X,Y)
                gradplus = self.cal_loss([X], [Y])
                parameter[ix] = original_value - h
                # gradminus = self.cal_total_loss(X,Y)
                gradminus = self.cal_loss([X], [Y])
                # 估计的梯度
                estimated_gradient = (gradplus - gradminus) / (2 * h)
                # 将参数重置为原来的值
                parameter[ix] = original_value
                # 用反向传播计算参数的梯度
                backprop_gradinet = bptt_gradients[pIdx][ix]
                # 计算相关误差error：(|x-y|/(|x|+|y|))
                relative_error = np.abs(backprop_gradinet - estimated_gradient) / (np.abs(backprop_gradinet) + np.abs(estimated_gradient))
                # 如果误差过大，大于误差阈值，输出各参数值以检查
                if(relative_error > error_threshold):
                    print('梯度检查错误：parameter=%s,ix=%s' % (pName,pIdx))
                    print('+h Loss:%f' % gradplus)
                    print('-h Loss:%f' % gradminus)
                    print('estimated_gradient:%lf' % estimated_gradient)
                    print('backpropagation gradient:%lf' % backprop_gradinet)
                    print('relative error:%lf' % relative_error)
                    return
            it.iternext()  # 进入下一次迭代
        print('参数 %s 的梯度检查通过' % (pName))
    # end of gradient_check

    def numpy_sgd_step(self,X,Y,learning_rate):
        """
        更新参数U、V、W
        :param X: 单个样本的输入
        :param Y: 单个样本的输入
        :param learning_rate: 学习率
        """
        # 计算梯度
        dU,dV,dW = self.bptt(X, Y)
        # 更新参数U、V、W
        self.U -= learning_rate * dU
        self.V -= learning_rate * dV
        self.W -= learning_rate * dW
    # end of numpy_sgd_step

    def momentum(self,X,Y,learning_rate,VdU,VdV,VdW,beta=0.9):
        """
        momentum方法更新参数U、V、W
        :param X: 单个样本的输入
        :param Y: 单个样本的输入
        :param learning_rate: 学习率
        """
        # 计算梯度
        dU,dV,dW = self.bptt(X, Y)
        VdU = beta * VdU + (1 - beta) * dU
        VdV = beta * VdV + (1 - beta) * dV
        VdW = beta * VdW + (1 - beta) * dW
        # 更新参数U、V、W
        self.U -= learning_rate * VdU
        self.V -= learning_rate * VdV
        self.W -= learning_rate * VdW
        # print('U:',self.U)
        return VdU, VdV, VdW
    # end of momentum
# end of class RNNNumpy

def train_sgd(model, X_train, Y_train, learing_rate=0.05, epoch=100, evaluate_loss_after=5):
    """
    使用SGD（随机梯度下降）训练
    :param model: RNN模型
    :param X_train: 训练数据输入
    :param Y_train: 训练数据输出
    :param learing_rate: 学习率
    :param epoch:  迭代完整数据集的次数
    :param evaluate_loss_after: 在多少次epoch后估计损失
    """
    losses = []
    num_examples_seen = 0
    VdU = np.zeros(model.U.shape)
    VdV = np.zeros(model.V.shape)
    VdW = np.zeros(model.W.shape)
    log_data = []
    for i in range(epoch):
        if(i % evaluate_loss_after ==0):
            loss = model.cal_loss(X_train, Y_train)
            losses.append((num_examples_seen, loss))
            time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            print('%s: loss after num_examples_seen=%d,epoch=%d:%lf' % (time,num_examples_seen,i,loss))
            # 如果loss增加了，调整learning_rate（学习率衰减）
            if(len(losses)>1 and losses[-1][1]>losses[-2][1]):
                learing_rate *= 0.25
                print('设置learning_rate为：%lf' % learing_rate)
            sys.stdout.flush()  # 刷新stdout，实时输出
        # 循环训练集
        for i in range(len(Y_train)):
            # model.numpy_sgd_step(X_train[i],Y_train[i],learing_rate)
            VdU,VdV,VdW = model.momentum(X_train[i],Y_train[i],
                                         learing_rate,VdU,VdV,VdW,beta=0.99)
            # print('VdU:',VdU)
            num_examples_seen += 1
        log_data.append([learing_rate,loss])
    saveLossLog.log_save(log_data)
# end of train_sgd

train_x, train_y, vocabulary_size, index_to_word, tokenized_sentences,keywords,flags = pretreatText.dataset()

np.random.seed(3)
# 构建模型类
# model1 = RNNNumpy(vocabulary_size, hidden_dim=225)
model = RNNNumpy(vocabulary_size, hidden_dim=225)
# o,s = model.forward_propagation(train_x[0])
# print('o.shape:', o.shape)
# print('o:', o)

# predictions = model.predict_with_tfidf(train_x[:2])
# print('predictions:',predictions[1][58],' len:',len(predictions))

tfidf_dict = tfidfRNN.tfidf()  # 获得词到TF-IDF权重
# 20% predict + POS + 50% tf-idf
def output_keys_tfidf(model,train_x,tokenized_sentences,tfidf_dict,T_RATE,flags):
    predictions = model.predict_with_tfidf(train_x)  # 得到在词表中的标index
    word = []
    keys = []
    predict_kw = []
    pos_list = ['n','np','ns','ni','nz','j','i','m','v']
    # print('predictions:', predictions, ' len:', len(predictions))
    # print('tokenized_sentences:',tokenized_sentences)
    for i, predict in enumerate(predictions):
        # print('i:', i,' predict:',predict)
        for value in predict:  # value is like (idx,possibility)
            if(flags[i][value[0]] in pos_list):
                # print('i:',i,',value[0]:',value[0],',flags[i][value[0]]:',flags[i][value[0]])
                word.append(tokenized_sentences[i][value[0]])
                # word = list(set(word))  # 排除重复词
        keys.append(word)
        word = []
    # print('keys1:', keys, ' len:', len(keys))
    for i,words in enumerate(keys):
        temp = []
        temp1 = []
        for w in words:
            # print('w:',w)
            if(w[0] in tfidf_dict[i].keys()):
                temp.append((w,tfidf_dict[i][w[0]]))
            # print('temp:', temp)
        if(len(words)<T_RATE):
            topK = len(words)
        else:
            topK = len(words) // T_RATE
        # print('topK:',topK)
        temp_k = sorted(temp,key=lambda w:w[1],reverse=True)[:topK]
        for tuple in temp_k:
            temp1.append(tuple[0])
        predict_kw.append(temp1)
        # print('temp_k:',temp_k)
    # print('keys:', predict_kw, ' len:', len(predict_kw))
    # 去重复词汇
    predict_kws = []
    for words in predict_kw:
        temp_w = [w[0] for w in words]
        predict_kws.append(set(temp_w))
    print('keys:', predict_kws, ' len:', len(predict_kws))
    return predict_kw
# end of output_keys_tfidf

def test(model,train_x,tokenized_sentences):
    # predictions = model.predict_with_tfidf(train_x)  # 得到在词表中的标index
    # if(len(predictions)==len(tokenized_sentences)):
    #     print('len equal')
    #     for i in np.arange(len(predictions)):
    #         maxV = max(value[0] for value in predictions[i])
    #         if(maxV>len(tokenized_sentences[i])):
    #             print('idx:%d out of range',i,' max:',maxV)
    #         else:
    #             print('pass')

    # for i in np.arange(len(train_x)):
    #     if(len(train_x[i])==len(tokenized_sentences[i])):
    #         print('train x equal:',len(train_x[i]))
    #     else:
    #         print('not equal,i:',i,' len(train_x[i])',len(train_x[i]),
    #               ' len(tokenized_sentences[i])',len(tokenized_sentences[i]))

    for i in np.arange(len(train_x)):
        o, s = model.forward_propagation(train_x[i])
        if(len(o)==len(tokenized_sentences[i])):
            print('o equal:', len(o))
        else:
            print('not equal,i:', i, ' len(o)', len(o),
                  ' len(tokenized_sentences[i])',len(tokenized_sentences[i]))

def test2(flags,train_x):
    predictions = model.predict_with_tfidf(train_x)
    for i, predict in enumerate(predictions):
        # print('i:', i,' predict:',predict)
        for value in predict:
            # print('i:', i,'value[0]:',value[0])
            if(value[0]>len(flags[i])):
                print('i:', i,'len flags:',len(flags[i]),'value[0]:',value[0])
            else:
                print('flag:',flags[i][value[0]])
            #     print('len flags:', len(flags[i]))
    # print('train_x len:',train_x[1])

sample_train_idx = len(train_x) // 10 * 7  # 70% train
sample_eva_idx = len(train_x)-sample_train_idx  # 30% evaluate
print('sample_train_num:', sample_train_idx, ' sample_eva_num:', sample_eva_idx)

# load model
infile = '/home/ubuntu/文档/models/05250130-momentum-hidden225.npz'
model.U,model.V,model.W,model_train_x = load_model_parameters(infile)
model.hidden_dim = model.U.shape[0]
model.word_dim = model.U.shape[1]
# 梯度检查测试
# for i in np.arange(4):
#     model1.gradient_check(train_x[i], train_y[i])

# test
# predictions = model1.predict_with_tfidf(train_x[2])
# print('predictions:',predictions,' len:',len(predictions))
# print('train_y:',train_y[:2])
# print('train_x:',train_x[0])

# train
# losses = train_sgd(model, model_train_x[:sample_train_idx], train_y[:sample_train_idx],
#                    learing_rate=0.00001, epoch=1000, evaluate_loss_after=1)
# 保存model
# outfile = '/home/ubuntu/文档/models/05251420-momentum-hidden225.npz'
# save_model_parameters(outfile,model,train_x)

# train data set
# output_keys_tfidf(model,train_x[:sample_train_idx],tokenized_sentences[:sample_train_idx],tfidf_dict[:sample_train_idx],T_RATE)
# evaluate data set
predict_kw = output_keys_tfidf(model,train_x[sample_train_idx:],tokenized_sentences[sample_train_idx:],tfidf_dict[sample_train_idx:],T_RATE,flags[sample_train_idx:])

# evaluate precision & recall
evaluate.evaluate_rnn(keywords[sample_train_idx:],predict_kw)
#
# test(model,train_x,tokenized_sentences)
# test2(flags,train_x[sample_train_idx:])

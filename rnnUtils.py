# -*- coding: utf-8 -*-
# 用于RNN的辅助函数
import numpy as np

def softmax(x):
    xt = np.exp(x - np.max(x))
    # xt = np.exp(x)
    return xt / np.sum(xt)

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def save_model_parameters(outfile, model,train_x):
    # U, V, W = model.U.get_value(), model.V.get_value(), model.W.get_value()
    U, V, W = model.U, model.V, model.W
    np.savez(outfile, U=model.U, V=model.V, W=model.W, x=train_x)
    print("Saved model parameters to %s." % outfile)

def load_model_parameters(path):
    npzfile = np.load(path)
    U, V, W = npzfile["U"], npzfile["V"], npzfile["W"]
    train_x = npzfile['x']
    # model.hidden_dim = U.shape[0]
    # model.word_dim = U.shape[1]
    # model.U = U
    # model.V = V
    # model.W = W
    print("Loaded model parameters from %s. hidden_dim=%d word_dim=%d" % (path, U.shape[0], U.shape[1]))
    return U, V, W,train_x
# -*- coding: utf-8 -*-
"""
@author: inversion
"""

import pandas as pd
import numpy as np

from sklearn.cross_validation import train_test_split
from sklearn.cross_decomposition import PLSRegression


def multiclass_log_loss(y_true, y_pred, eps=1e-15):
    """Multi class version of Logarithmic Loss metric.
    https://www.kaggle.com/wiki/MultiClassLogLoss

    from:
    https://www.kaggle.com/c/predict-closed-questions-on-stack-overflow/forums/t/2644/multi-class-log-loss-function

    Parameters
    ----------
    y_true : array, shape = [n_samples]
    y_pred : array, shape = [n_samples, n_classes]

    Returns
    -------
    loss : float
    """
    predictions = np.clip(y_pred, eps, 1 - eps)

    # normalize row sums to 1
    predictions /= predictions.sum(axis=1)[:, np.newaxis]

    actual = np.zeros(y_pred.shape)
    rows = actual.shape[0]
    actual[np.arange(rows), y_true.astype(int)] = 1
    vsota = np.sum(actual * np.log(predictions))
    return -1.0 / rows * vsota



train = pd.read_csv('train.csv', index_col='id')
targets = pd.get_dummies(train.target)
train.drop('target', axis=1, inplace=True)
train = train.apply(np.log1p)

test = pd.read_csv('test.csv', index_col='id')
test = test.apply(np.log1p)

Xt, Xv, yt, yv = train_test_split(train, targets, test_size=0.2, random_state=27)

best = 10.

for n in range(5,16):
    
    clf = PLSRegression(n_components=n)
    clf.fit(Xt,yt)
    y_pred = clf.predict(Xv)
    loss = multiclass_log_loss(np.argmax(y_pred,axis=1),y_pred)
    if loss < best:
        n_best = n
        best = loss
        postfix = '(*)'
    else:
        postfix = ''
    print ('comps: {:02d}\tLoss:{:5.4f} {}'.format(n,loss,postfix))


clf = PLSRegression(n_components=n_best)  
clf.fit(train,targets)
y_pred = clf.predict(test)


submission = pd.read_csv('sampleSubmission.csv')
submission.set_index('id', inplace=True)
submission[:] = y_pred
submission.to_csv('benchmark_pls.csv')




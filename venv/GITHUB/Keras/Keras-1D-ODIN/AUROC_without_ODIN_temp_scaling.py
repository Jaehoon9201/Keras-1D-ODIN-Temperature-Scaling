from itertools import cycle

from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical
from keras.engine.saving import load_model
from keras.layers.convolutional import UpSampling2D, Conv1D

from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import auc

import seaborn as sns
import math
import pandas as pd
import numpy as np
import glob
import os
import matplotlib.pyplot as plt
from collections import Counter

#■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■
#■■■■■■■■■■■■■■■■■■■ ROC & AUC ■■■■■■■■■■■■■■■■■■■■
#■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■

n_classes = 5
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(0, n_classes) : # 0에서 10까지

    df2 = pd.read_csv('data/train_data.csv')
    err_train = df2.loc[:,
                ['data1', 'data2', 'data3', 'data4', 'data5', 'data6', 'data7', 'data8', 'data9', 'data10', 'data11',
                 'data12', 'data13', 'data14']]
    status_train = df2.values[:, -1].astype(np.int64)

    df3 = pd.read_csv('data/test_data_with_OOD.csv')
    err_test = df3.loc[:,
               ['data1', 'data2', 'data3', 'data4', 'data5', 'data6', 'data7', 'data8', 'data9', 'data10', 'data11',
                'data12', 'data13', 'data14']]
    status_test = df3.values[:, -1].astype(np.int64)


    print(Counter(status_train))
    status_train[(status_train != i)] = -2
    status_train[(status_train == i)] = -1
    status_train[(status_train == -2)] = 0
    status_train[(status_train == -1)] = 1
    print(Counter(status_train))

    print(Counter(status_test))
    status_test[(status_test != i)] = -2
    status_test[(status_test == i)] = -1
    status_test[(status_test == -2)] = 0
    status_test[(status_test == -1)] = 1
    print(Counter(status_test))


    model = load_model('model/ acti_relu train_epoch batch_size = 2000 1024/1921 0.06861.hdf5'.format(i))

    # ROC & AUC
    test_pred = model.predict(err_test)
    print('---test_pred2----')
    print(np.shape(test_pred))

    test_pred1 = test_pred[:, i]
    if i == 0 :
        test_pred2 = np.sum(test_pred[:, 1:5]                       ,axis = 1)
    elif i == 1 :
        test_pred2 = np.sum(test_pred[:, 2:5]    ,axis = 1)
        test_pred2 = test_pred2.reshape(len(test_pred2),1)
        test_pred22 = np.sum(test_pred[:, 0:1]   ,axis = 1)
        test_pred22 = test_pred22.reshape(len(test_pred22), 1)
        test_pred2 = np.sum(test_pred2+ test_pred22 ,axis = 1)
    elif i == 2 :
        test_pred2 = np.sum(test_pred[:, 3:5]    ,axis = 1)
        test_pred2 = test_pred2.reshape(len(test_pred2),1)
        test_pred22 = np.sum(test_pred[:, 0:2]   ,axis = 1)
        test_pred22 = test_pred22.reshape(len(test_pred22), 1)
        test_pred2 = np.sum(test_pred2+ test_pred22 ,axis = 1)
    elif i == 3 :
        test_pred2 = np.sum(test_pred[:, 4:5]    ,axis = 1)
        test_pred2 = test_pred2.reshape(len(test_pred2),1)
        test_pred22 = np.sum(test_pred[:, 0:3]   ,axis = 1)
        test_pred22 = test_pred22.reshape(len(test_pred22), 1)
        test_pred2 = np.sum(test_pred2+ test_pred22 ,axis = 1)
    elif i == 4 :
        test_pred2 = np.sum(                    test_pred[:, 0:4]   ,axis = 1)

    test_pred1 = test_pred1.reshape(len(test_pred2),1)
    test_pred2 = test_pred2.reshape(len(test_pred2),1)
    test_pred = np.concatenate((test_pred1, test_pred2), axis = 1)
    #test_pred = test_pred.ravel()

    status_test = status_test.reshape(len(status_test),1)
    #status_test = status_test.ravel()

    print('---check----')
    print(np.shape(status_test))
    print(status_test)
    print(test_pred1)

    fpr[i], tpr[i], thresholds = roc_curve(status_test.ravel(), test_pred1.ravel())

    roc_auc[i] = auc(fpr[i], tpr[i])

    plt.figure(n_classes)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.plot(fpr[i], tpr[i], label='Fault [{:.0f}] AUC = {:.4f}'.format(i, roc_auc[i]))
    print('---check----')
    print(fpr[i])




# First aggregate all false positive rates
all_fpr = np.unique(np.concatenate([fpr[j] for j in range(0, n_classes )]))
# Then interpolate all ROC curves at this points
mean_tpr = np.zeros_like(all_fpr)
for k in range(0, n_classes ):
    mean_tpr += np.interp(all_fpr, fpr[k], tpr[k])
mean_tpr /= (n_classes )

fpr["macro"] = all_fpr
tpr["macro"] = mean_tpr
roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
plt.plot(fpr["macro"], tpr["macro"],
         label='macro-avg AUC = {:.4f}'.format(roc_auc["macro"]),
         color='navy', linestyle=':', linewidth=4)
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('Fault ROC curve')
plt.legend(loc='best')


plt.show()



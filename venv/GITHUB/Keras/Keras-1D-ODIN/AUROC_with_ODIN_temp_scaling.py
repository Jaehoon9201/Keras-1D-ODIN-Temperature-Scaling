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
from tensorflow.python.keras import Input
from tensorflow.python.keras.backend import categorical_crossentropy
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.utils.np_utils import to_categorical
from keras import backend as K
from itertools import repeat
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from collections import Counter
import tensorflow as tf

#■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■
#■■■■■■■■■■■■■■■■■■■ ROC & AUC ■■■■■■■■■■■■■■■■■■■■
#■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■

n_classes = 5
epsilon = 0.1
noiseMagnitude1 = epsilon


fpr = dict()
tpr = dict()
roc_auc = dict()

f3 = open("./T_scaling_results/T_scaling_results.txt", 'w')

#var = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900, 2000, 2100, 2200, 2300, 2400, 2500, 2600, 2700, 2800, 2900, 3000, 3100, 3200, 3300, 3400, 3500, 3600, 3700, 3800, 3900, 4000, 4100, 4200, 4300, 4400, 4500,4600, 4700,4800,4900,5000,5100,5200,5300,5400,5500,5600,5700,5800,5900,6000,6100,6200,6300,6400,6500,6600,6700,6800,6900,7000,7100,7200,7300,7400,7500,7600,7700,7800,7900,8000,8100,8200,8300,8400,8500,8600,8700,8800,8900,9000,9100,9200,9300,9400,9500,9600,9700,9800,9900,10000]
#var = [4200]
var = [7900]

for j in range(len(var)):
    temper = var[j]

    for i in range(0, n_classes) : # 0에서 10까지

        df2 = pd.read_csv('data/train_data.csv')
        err_train = df2.loc[:,
                    ['data1', 'data2', 'data3', 'data4', 'data5', 'data6', 'data7', 'data8', 'data9', 'data10',
                     'data11',
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


        print('---test_pred----')
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
            test_pred2 = np.sum( test_pred[:, 0:4]   ,axis = 1)

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



        # ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■
        # ■■■■■■■■■■■■■■■■■ ODIN ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■
        # ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■



        test_classes = 2
        input_shape = 14
        delta = 1e-1
        err_test_del = np.zeros(14*len(err_test)).reshape((len(err_test), 14))

        gradient = np.zeros(test_classes*len(err_test)).reshape((len(err_test), test_classes))
        gradient_temp = np.zeros(test_classes*len(err_test) * 14).reshape((len(err_test), test_classes, 14))
        gradient_tot = np.zeros(test_classes*len(err_test)*2**14).reshape((len(err_test), test_classes, 2**14))
        test_pred_del = np.zeros(test_classes*len(err_test)).reshape((len(err_test), test_classes))
        nnOutputs_del = np.zeros(test_classes*len(err_test)).reshape((len(err_test),test_classes))

        test_pred = model.predict(err_test)
        #test_pred_in = model.predict(err_test_in)
        #test_pred_out = model.predict(err_test_out)

        labels = to_categorical(status_test)
        labels_tensor = tf.multiply(labels, 1)  # tf.cast(labels, tf.float32)
        y_labels = labels[0:len(status_test)]


        # ■■■■■■■■■■■■■■ Input Processing ■■■■■■■■■■■■■■■
        #Adding small perturbations to images
        outputs = model.predict(err_test)
       # outputs_in = model.predict(err_test_in)
       # outputs_out = model.predict(err_test_out)
        # ■■■■■■■■■■ Temperature scailing parameter■■■■■■■■■■■
        outputs = outputs / temper
        #outputs_in = outputs_in / temper
        #outputs_out = outputs_out / temper
        print('--- outputs check----')
        print(outputs.shape)
        print(outputs)
        # Calculating the confidence atfer adding perturbations
        # ■■■■■■■■■■■■■■■Softmax 함수 구현 ■■■■■■■■■■■■■■■
        outputs = np.exp(outputs) / np.expand_dims(np.sum(np.exp(outputs), axis=1), 1)
        outputs = outputs[:, i]
        # outputs_in = np.exp(outputs_in) / np.expand_dims(np.sum(np.exp(outputs_in), axis=1), 1)
        # outputs_in = outputs_in[:, i]
        # outputs_out = np.exp(outputs_out) / np.expand_dims(np.sum(np.exp(outputs_out), axis=1), 1)
        # outputs_out = outputs_out[:, i]
        #f1.write("{}, {}, {}\n".format(temper, noiseMagnitude1, np.max((outputs_in), axis = 0)))
        #f2.write("{}, {}, {}\n".format(temper, noiseMagnitude1, np.max((outputs_out), axis = 0)))

        print('--- nnOutputs Ravel check----')
        print(outputs.ravel().shape)
        print(outputs)


        fpr[i], tpr[i], thresholds = roc_curve(status_test.ravel(), outputs.ravel())
        print('---thresholds----')
        print(thresholds)
        roc_auc[i] = auc(fpr[i], tpr[i])

        plt.figure(n_classes)
        plt.plot([0, 1], [0, 1], 'k--')
        plt.plot(fpr[i], tpr[i], label='Fault [{:.0f}] AUC = {:.4f}'.format(i, roc_auc[i]))
        print('---check----')
        print(fpr[i])



    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[j] for j in range(0, n_classes)]))
    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for k in range(0, n_classes):
        mean_tpr += np.interp(all_fpr, fpr[k], tpr[k])
    mean_tpr /= (n_classes)

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
    f3.write( "Temperature scale {:20}      ROC_AUC macro{:13.6f}\n".format(temper, roc_auc["macro"]))



f3.close()






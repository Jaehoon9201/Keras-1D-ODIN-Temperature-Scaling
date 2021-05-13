from keras.datasets import mnist
from sklearn.metrics import classification_report, precision_score, recall_score, f1_score, confusion_matrix
from tensorflow.keras import layers
import tensorflow as tf
from tensorflow.python.keras.losses import binary_crossentropy
import keras as K
import keras
import numpy as np
from tensorflow.python.keras.utils.np_utils import to_categorical
import matplotlib.pyplot as plt
import seaborn as sns
from keras.layers import Input, Dense
from keras.models import Model
import pandas as pd
import numpy as np
import glob
import os
from sklearn.utils import shuffle
from sklearn.preprocessing import MinMaxScaler
from keras.callbacks import ModelCheckpoint, EarlyStopping


#■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■
#■■■■■■■■■■■■■■■ Classification Train ■■■■■■■■■■■■■■■■■
#■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■


df2 = pd.read_csv('data/train_data.csv')
err_train = df2.loc[:,['data1','data2', 'data3', 'data4', 'data5', 'data6', 'data7', 'data8', 'data9', 'data10', 'data11', 'data12', 'data13', 'data14']]
status_train = df2.values[:, -1].astype(np.int64)

df3 = pd.read_csv('data/test_data.csv')
err_test = df3.loc[:,['data1','data2', 'data3', 'data4', 'data5', 'data6', 'data7', 'data8', 'data9', 'data10', 'data11', 'data12', 'data13', 'data14']]
status_test = df3.values[:, -1].astype(np.int64)

print(err_train)
print(np.shape(err_train))
print(np.shape(status_train))
print(np.shape(err_test))
print(np.shape(status_test))
status_train_cat = to_categorical(status_train)
status_test_cat = to_categorical(status_test)



original_dim = 14
dense1 = 16
dense2 = 16

# ■■■■■■■ Load model and eval ■■■■■■■■

#var = [40, 80, 120, 240, 360, 720, 1080, 1800, 2520, 3960, 5400, 8280, 11160,16920,22680, 34200, 45720]
var = [512, 1024]
for i in range(len(var)):
    var_str = 'acti_relu train_epoch batch_size'
    var1 =2000
    var2 = var[i]
    train_epoch = var1
    batch_size = var2

    class_num = 5
    learn_rate = 2e-5

    # ■■■■■■■■■■■■■■■■■■
    # ■■■■■■   NN   ■■■■■■■
    # ■■■■■■■■■■■■■■■■■■

    # NN Network with classification output only
    # Single task classification model
    sc_input_img=Input(shape=(original_dim,), name='input')
    x = Dense(dense1, activation='relu', name='nn_1')(sc_input_img)
    x = Dense(dense2, activation='relu', name='nn_2')(x)
    SC = Dense(class_num, activation='softmax', name='SC')(x)
    # Take input and give classification and reconstruction
    NN_SC = Model(sc_input_img, SC)

    optimizer = tf.keras.optimizers.Adam(learning_rate=learn_rate)
    NN_SC.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])
    NN_SC.summary()

    MODEL_SAVE_FOLDER_PATH = './model/ %s = %d %d/' % (var_str , var1, var2)
    if not os.path.exists(MODEL_SAVE_FOLDER_PATH):
      os.mkdir(MODEL_SAVE_FOLDER_PATH)
    model_path = MODEL_SAVE_FOLDER_PATH + '{epoch:04d} {val_loss:.5f}.hdf5'

    checkpoint = ModelCheckpoint(filepath=model_path, monitor='val_loss', verbose=1,
                                 save_best_only=True, mode='auto', period=1)
    earlystopping = EarlyStopping(monitor='val_loss', patience=140 )
    # Single-Task Train
    SC_history = NN_SC.fit(err_train, status_train_cat,
                                          epochs=train_epoch, batch_size=batch_size, shuffle=True, verbose=1,
                                          validation_data=(err_test, status_test_cat), callbacks = [checkpoint, earlystopping])

    #■■■■■■■ Learning saving  ■■■■■■■■

    fig1 = plt.figure(1)
    plt.plot(SC_history.history['loss'], label='Train NN')
    plt.plot(SC_history.history['val_loss'], label='Test NN')
    plt.title('Model loss %s = %d %d'% (var_str , var1, var2))
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend()
    figname1 = 'Model loss %s = %d %d.png'% (var_str , var1, var2)
    model_path1 = MODEL_SAVE_FOLDER_PATH + figname1
    plt.savefig(model_path1)
    plt.close(fig1)


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

from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score, f1_score
import seaborn as sns
import pandas as pd
import numpy as np
from keras.engine.saving import load_model
import matplotlib.pyplot as plt
import tensorflow as tf
from keras import backend as K
#■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■
#■■■■■■■■■■■■■■■■ Classification Test ■■■■■■■■■■■■■■■■■
#■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■


model = load_model('model/ acti_relu train_epoch batch_size = 2000 1024/1921 0.06861.hdf5')

df_test2 = pd.read_csv('data/test_data.csv')
err_test = df_test2.loc[:,
           ['data1', 'data2', 'data3', 'data4', 'data5', 'data6', 'data7', 'data8', 'data9', 'data10', 'data11',
            'data12', 'data13', 'data14']]
status_test = df_test2.values[:, -1].astype(np.int64)


test_pred = np.argmax(model.predict(err_test), axis=1)

# ■■■■■ specific layer out ■■■■■
get_1st_layer_output = K.function([model.layers[0].input],
                                  [model.layers[1].output])
layer_output1 = get_1st_layer_output([err_test])[0]
print(layer_output1)

print(type(layer_output1))
get_2st_layer_output = K.function([model.layers[0].input],
                                  [model.layers[2].output])
layer_output2 = get_2st_layer_output([err_test])[0]
print(layer_output2)
print(type(layer_output2))
# ■■■■■■■■■■■■■■■■■■■■■■

print(model.predict(err_test))
print(type(model.predict(err_test)))

print(classification_report(status_test, test_pred, digits = 4))

print("PRE: %.4f" % precision_score(status_test, test_pred, average='micro'))
print("REC: %.4f" % recall_score(status_test, test_pred, average='micro'))
print("F1: %.4f" % f1_score(status_test, test_pred, average='micro'))

confmat = confusion_matrix(y_true=status_test, y_pred=test_pred)
sns.heatmap(confmat, annot = True, fmt ='d',cmap = 'Blues')
plt.xlabel('predicted label')
plt.ylabel('true label')
plt.show()








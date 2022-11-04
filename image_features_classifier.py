
import os
import sys
import shutil
from PIL import Image
import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import Dense, Flatten, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions


from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten, GlobalAveragePooling2D
import matplotlib.pyplot as plt
from itertools import cycle

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import label_binarize
from sklearn import preprocessing

#from sklearn import svm, datasets
from sklearn.metrics import roc_curve, auc
#from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
skf = StratifiedKFold(n_splits=5)

x_img_ftr = xx_img
y_targ = img_stg

y_targ = np.array(y_targ, dtype="uint8")

num_classes = len(np.unique(y_targ))

y_targ_categorical = keras.utils.to_categorical(y_targ, num_classes)


#Splitting each of the three dataset into 80%training 20%test
for i, (train_index, test_index) in enumerate(skf.split(x_img_ftr, y_targ_categorical.argmax(1))):
    x_img_train, x_img_test = x_img_ftr[train_index], x_img_ftr[test_index]
    y_train, y_test = y_targ[train_index], y_targ[test_index]


y_train_categorical = keras.utils.to_categorical(y_train, num_classes)


fold_no = 1
acc_per_fold_img = []
loss_per_fold_img = []
xx_img_tra = []
xx_img_tes = []
yy_img_tra = []
yy_img_tes = []

for i, (train_index, val_index) in enumerate(skf.split(x_img_train, y_train_categorical.argmax(1))):
    
    x_img_train_kf, x_img_val_kf = x_img_train[train_index], x_img_train[val_index]
    
    y_train_kf, y_val_kf = y_train[train_index], y_train[val_index]
    
    xx_img_tra.append(x_img_train_kf)
    xx_img_tes.append(x_img_val_kf)
    yy_img_tra.append(y_train_kf)
    yy_img_tes.append(y_val_kf)
    
    y_train_k = keras.utils.to_categorical(y_train_kf, 4)
    y_val_k = keras.utils.to_categorical(y_val_kf, 4)
    
    
    
    #img
    inputs = keras.Input(shape=(2048,))
    #x = GlobalAveragePooling2D()(x)    
    #inputs_img = keras.Input(shape=(100352,))
    x = layers.Dense(1024, activation="relu")(inputs)
    outputs = layers.Dense(4, activation="softmax")(x)
    model_img = keras.Model(inputs, outputs)
  
    model_img.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

       
    # Generate a print
    print('------------------------------------------------------------------------')
    print(f'Training for fold {fold_no} ...')
    
    history_img = model_img.fit(x_img_train_kf, y_train_k, 
                epochs=200,
                shuffle=True,
                validation_data = (x_img_val_kf, y_val_k))

    scores_img = model_img.evaluate(x_img_val_kf, y_val_k, verbose=0)


    print(f'Score for fold {fold_no}: {model_img.metrics_names[0]} of {scores_img[0]}; {model_img.metrics_names[1]} of {scores_img[1]*100}%')
    acc_per_fold_img.append(scores_img[1] * 100)
    loss_per_fold_img.append(scores_img[0])

    
    # Increase fold number
    fold_no = fold_no + 1


#Ploting the model accuracy
history = history_img

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()


#Generating the confusion matrix for images
Pred_img= model_img.predict(x_img_test)
Pred_Label_img = np.argmax(Pred_img, axis=1)

cm_img = tf.math.confusion_matrix(y_test, Pred_Label_img)

#Generating AUC-ROC for images
def multiclass_roc_auc_scoreFD(y_test, y_pred, average="macro"):
    targ_stg= [0, 1, 2, 3]
    fig, c_ax = plt.subplots(1,1, figsize = (12, 8))
    lb = preprocessing.LabelBinarizer()
    lb.fit(y_test)
    y_test = lb.transform(y_test)
    y_pred = lb.transform(y_pred)

    for (idx, c_label) in enumerate(targ_stg):
        fpr, tpr, thresholds = roc_curve(y_test[:,idx].astype(int), y_pred[:,idx])
        c_ax.plot(fpr, tpr, label = '%s (AUC:%0.2f)'  % (c_label, auc(fpr, tpr)))
    c_ax.plot(fpr, fpr, 'b-', label = 'Random Guessing')
    c_ax.legend()
    c_ax.set_xlabel('False Positive Rate')
    c_ax.set_ylabel('True Positive Rate')
    p = plt.show()
    return roc_auc_score(y_test, y_pred, average=average), p


#Determining the test accuracy
ytest = keras.utils.to_categorical(y_test, 4)
test_acc_img = model_img.evaluate(x_img_test, ytest, verbose=0)
test_acc_img

#Estimating the validation accuracy
validation_acc_img = sum(acc_per_fold_img)/len(acc_per_fold_img)


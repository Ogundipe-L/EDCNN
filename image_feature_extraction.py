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


dir_pline = 'image tiles directory'
dfclin = pd.read_table("clinical data directory")


# we chose to train the top 3 ResNet50 blocks, i.e. we will freeze
# the first 172 layers and unfreeze the rest:
for layer in model.layers[:172]:
   layer.trainable = False
for layer in model.layers[172:]:
   layer.trainable = True

# we need to recompile the model for these modifications to take effect
# we use SGD with a low learning rate
from tensorflow.keras.optimizers import SGD
model.compile(optimizer=SGD(learning_rate=0.0001, momentum=0.9), loss='categorical_crossentropy', metrics=["accuracy"])


# train the model on the new data for a few epochs
epochs = 40

model.fit(
    train_ds, epochs=epochs, validation_data=val_ds,
)


model2 = Model(model.input, model.layers[-4].output)


def clin_stg_numr(clinc):
    p_id = (clinc["bcr_patient_barcode"]).tolist()
    p_stg = (clinc["ajcc_pathologic_tumor_stage"]).tolist()
    dfstg = []
    for i in range(len(p_id)):
        idf = p_id[i]
        stg = p_stg[i]
        if stg in ('Stage I', 'Stage IA'):
            sp_stg = 0
            dfstg.append(sp_stg)
        elif stg in  ('Stage II', 'Stage IIA', 'Stage IIB', 'Stage IIC'):
            sp_stg = 1
            dfstg.append(sp_stg)
        elif stg in  ('Stage III', 'Stage IIIA', 'Stage IIIB', 'Stage IIIC'):
            sp_stg = 2
            dfstg.append(sp_stg)
        elif stg in  ('Stage IV', 'Stage IVA', 'Stage IVB', 'Stage IVC'):
            sp_stg = 3
            dfstg.append(sp_stg)
                    
    return p_id, dfstg

def img_arry_stg(model, clinc, imgfl):
    from tensorflow.keras.applications.resnet50 import ResNet50
    from tensorflow.keras.preprocessing import image
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import Dense, Flatten, GlobalAveragePooling2D
    from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
    import numpy as np

        
    p_idf, stg_pd = clin_stg_numr(clinc)
    imgfn =[]
    img_arr = []
    img_arr_stg = []
    img_stg = []
    sbfn = []
    imgwofn = []
    x_y_z = []
    x_y = []
    img_flat_ftr = []
    xx_img = []
    ftr_img = []

    for fname in os.listdir(imgfl):
        ffnm = fname[17:29]
        if ffnm in p_idf:
            pid = p_idf.index(ffnm)
            stg = stg_pd[pid]
            img_stg.append(stg)
            imgfn.append(fname)
            sbfn.append(ffnm)
            
            img = image.load_img(fname, target_size=(224, 224))
            x = image.img_to_array(img)
            x = np.expand_dims(x, axis=0)
            x = preprocess_input(x)
            x = model.predict(x)
            x = GlobalAveragePooling2D()(x)
            xx_img.append(x)
        else:
            imgwofn.append(fname)
            
    return np.array(xx_img),sbfn, img_stg

os.chdir(dir_pline)

#Extracting features from image tiles using the pre-trained based model
xx_img, sbfn, img_stg = img_arry_stg(model2, dfclin, dir_pline1)
    
    return np.array(xx_img),sbfn, img_stg

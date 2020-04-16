#Evaluate the performance of the model for classification
#PASCAL VOC2009 --Soutrik

import keras
import numpy as np

model_path = "../model/saved_model.h5"
data_path= "../data/x_val.npy"

model = keras.models.load_model(model_path)

x_val = np.load(data_path)

print(model.evaluate(x_val,x_val,batch_size=32))


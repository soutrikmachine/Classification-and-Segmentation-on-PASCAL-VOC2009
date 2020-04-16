#Testing data 
#PASCAL VOC2009 

import numpy as np
import matplotlib
import matplotlib.pyplot as plot
matplotlib.use("TkAgg")

x_val = np.load("../data/x_val.npy")
y_val = np.load("../data/y_val.npy")


plot.imshow(x_val[10])
plot.show()

import numpy as np
import h5py
import scipy.io
import math

data = scipy.io.loadmat('S_tensors_2modes.mat')
print(data.keys())
SK = data['SK']
SR = data['SR']
np.set_printoptions(precision=10);
print(SK);
print(SK.flatten())
print(SR.flatten())




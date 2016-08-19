import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import os
import pickle

label = open('label.txt', 'r')
img_dir = 'train_face'
img_dirs = os.listdir(img_dir)
all = 1000
imgdict = {}
sum = np.zeros((224, 224, 3))
file = open('img_label.pkl', 'wb')
for i in range(all):
    line = label.readline()
    posspace = line.index(' ')
    imgname = line[0:posspace]
    imglabel = line[posspace+1 : -1]
    imglabel = int(imglabel)
    im = plt.imread(img_dir + '/' + imgname)
    im = np.array(im)
    if im.ndim == 2:
        sum[:, :, 0] += im
        sum[:, :, 1] += im
        sum[:, :, 2] += im
    else:
        sum = sum + im
    imgdict[imgname] = [im, imglabel]
pickle.dump(imgdict, file)
file.close()
img_avg = sum / all
file = open('imgavg.pkl', 'wb')
pickle.dump(img_avg, file)
file.close()
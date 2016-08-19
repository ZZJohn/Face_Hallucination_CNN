# -*- coding: utf-8 -*-
"""
Created on Mon Aug 15 17:16:30 2016

@author: zzy
"""

import caffe
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import savemat
import pickle

solver = caffe.SGDSolver('naive_solver.prototxt')
solver.net.copy_from('VGG_step_0.8_iter_400.caffemodel')

conv1_1_w = solver.net.params['conv1_1'][0].data[...]
conv1_2_w = solver.net.params['conv1_2'][0].data[...]
conv2_1_w = solver.net.params['conv2_1'][0].data[...]
conv2_2_w = solver.net.params['conv2_2'][0].data[...]
conv3_1_w = solver.net.params['conv3_1'][0].data[...]
conv3_2_w = solver.net.params['conv3_2'][0].data[...]
conv3_3_w = solver.net.params['conv3_3'][0].data[...]
conv4_1_w = solver.net.params['conv4_1'][0].data[...]
conv4_2_w = solver.net.params['conv4_2'][0].data[...]
conv4_3_w = solver.net.params['conv4_3'][0].data[...]
conv5_1_w = solver.net.params['conv5_1'][0].data[...]
conv5_2_w = solver.net.params['conv5_2'][0].data[...]
conv5_3_w = solver.net.params['conv5_3'][0].data[...]

conv1_1_b = solver.net.params['conv1_1'][1].data[...]
conv1_2_b = solver.net.params['conv1_2'][1].data[...]
conv2_1_b = solver.net.params['conv2_1'][1].data[...]
conv2_2_b = solver.net.params['conv2_2'][1].data[...]
conv3_1_b = solver.net.params['conv3_1'][1].data[...]
conv3_2_b = solver.net.params['conv3_2'][1].data[...]
conv3_3_b = solver.net.params['conv3_3'][1].data[...]
conv4_1_b = solver.net.params['conv4_1'][1].data[...]
conv4_2_b = solver.net.params['conv4_2'][1].data[...]
conv4_3_b = solver.net.params['conv4_3'][1].data[...]
conv5_1_b = solver.net.params['conv5_1'][1].data[...]
conv5_2_b = solver.net.params['conv5_2'][1].data[...]
conv5_3_b = solver.net.params['conv5_3'][1].data[...]

fc6_w = solver.net.params['fc6'][0].data[...]
fc7_w = solver.net.params['fc7'][0].data[...]
fc8_w = solver.net.params['fc8_finetune'][0].data[...]
fc6_b = solver.net.params['fc6'][1].data[...]
fc7_b = solver.net.params['fc7'][1].data[...]
fc8_b = solver.net.params['fc8_finetune'][1].data[...]

dic = {'conv1_1_w':conv1_1_w, 'conv1_2_w':conv1_2_w,
		'conv2_1_w':conv2_1_w, 'conv2_2_w':conv2_2_w,
		'conv3_1_w':conv3_1_w, 'conv3_2_w':conv3_2_w, 'conv3_3_w':conv3_3_w, 
		'conv4_1_w':conv4_1_w, 'conv4_2_w':conv4_2_w, 'conv4_3_w':conv4_3_w,
		'conv5_1_w':conv5_1_w, 'conv5_2_w':conv5_2_w, 'conv5_3_w':conv5_3_w,
		'conv1_1_b':conv1_1_b, 'conv1_2_b':conv1_2_b,
		'conv2_1_b':conv2_1_b, 'conv2_2_b':conv2_2_b,
		'conv3_1_b':conv3_1_b, 'conv3_2_b':conv3_2_b, 'conv3_3_b':conv3_3_b, 
		'conv4_1_b':conv4_1_b, 'conv4_2_b':conv4_2_b, 'conv4_3_b':conv4_3_b,
		'conv5_1_b':conv5_1_b, 'conv5_2_b':conv5_2_b, 'conv5_3_b':conv5_3_b,
		'fc6_w':fc6_w, 'fc7_w':fc7_w, 'fc8_w':fc8_w,
		'fc6_b':fc6_b, 'fc7_b':fc7_b, 'fc8_b':fc8_b}

#savemat('vgg_model', dic)
fp = open('weights_bin.pkl','wb')
pickle.dump(dic,fp,1)
fp.close()
import caffe
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy

#caffe.set_device(0)
#caffe.set_mode_gpu()
#solver = caffe.SGDSolver('bichannel_solver.prototxt')
#solver = caffe.SGDSolver('naive_solver.prototxt')
#solver.net.copy_from('VGG_FACE.caffemodel')
solver = caffe.SGDSolver('SRCNN_solver.prototxt')
solver.net.copy_from('SRCNN_iter_2500.caffemodel')

solver.solve()
'''solver.step(5)
#Iin = solver.net.blobs['Iin'].data[...]
#Iin = Iin[0,:,:,:]
ker1 = np.reshape(solver.net.params['conv1_1'][0].data[...],(64,3,9))+0.0
ker2 = np.reshape(solver.net.params['conv1_2'][0].data[...],(64,64,9))+0.0
ker3 = np.reshape(solver.net.params['conv2_1'][0].data[...],(128,64,9))+0.0
'''
'''solver.step(10)
Irec = solver.net.blobs['Irec'].data[...]
Irec = np.reshape(Irec, (200,3,100,100))
Irec = Irec[0,:,:,:]
Irec = np.transpose(Irec, (1,2,0))
Iin_flatten = solver.net.blobs['Iin_flatten'].data[...]
Iin_flatten = np.reshape(Iin_flatten, (200,3,100,100))
Iin_flatten = Iin_flatten[0,:,:,:]
Iin_flatten = np.transpose(Iin_flatten, (1,2,0))
Ih_flatten = solver.net.blobs['Ih_flatten'].data[...]
Ih_flatten = np.reshape(Ih_flatten, (200,3,100,100))
Ih_flatten = Ih_flatten[0,:,:,:]
Ih_flatten = np.transpose(Ih_flatten, (1,2,0))'''
#A = solver.net.blobs['a'].data[...]
#print 'A =', A[0:5]
'''fc22 = solver.net.params['full22'][0].data[...]-fc22
fc22_diff = solver.net.blobs['a'].diff[...]
fc22_bottom = solver.net.blobs['I5'].data[...]'''
#ker1 = np.reshape(solver.net.params['conv1_1'][0].data[...],(64,3,9))-ker1
#ker2 = np.reshape(solver.net.params['conv1_2'][0].data[...],(64,64,9))-ker2
#ker3 = np.reshape(solver.net.params['conv2_1'][0].data[...],(128,64,9))-ker3
#ker1 = np.reshape(solver.net.params['conv1'][0].data[...],(32,3,25))-ker1
#ker2 = np.reshape(solver.net.params['conv2'][0].data[...],(64,32,9))-ker2
#ker3 = np.reshape(solver.net.params['conv3'][0].data[...],(128,64,9))-ker3
'''Iin = solver.net.blobs['I2'].data[...]
Iin = np.reshape(Iin, (20,80,80))
Iin0 = Iin[0,:,:]
#Iin0 = np.transpose(Iin0, (1,2,0))
Iin1 = Iin[1,:,:]
#Iin1 = np.transpose(Iin1, (1,2,0))
Iin2 = Iin[2,:,:]
#Iin2 = np.transpose(Iin2, (1,2,0))'''
'''fig = plt.figure()
ax = fig.add_subplot(131)
ax.imshow(Ih_flatten)
ax = fig.add_subplot(132)
ax.imshow(Irec)
ax = fig.add_subplot(133)
ax.imshow(Iin_flatten)
plt.show()
solver.step(2555555)'''
'''for i in range(100):
    solver.step(5)
    I5 = solver.net.blobs['I5'].data
    fc22w = solver.net.params['full22'][0].data
    fc22b = solver.net.params['full22'][1].data'''
import caffe
import numpy as np

class MyLayer(caffe.Layer):

    def setup(self, bottom, top):
        # check input pair
        if len(bottom) != 3:
            raise Exception("Need three inputs to compute linear combination.")

    def reshape(self, bottom, top):
        # check input dimensions match-
        if bottom[0].count != bottom[2].count:
            raise Exception("Inputs must have the same dimension."+str(bottom[0].data.shape)
            +"\n"+str(bottom[2].data.shape))
        top[0].reshape(*bottom[0].data.shape)

    def forward(self, bottom, top):
        top[0].data[...] = np.multiply(bottom[0].data.T, 1-bottom[1].data.T).T \
                         + np.multiply(bottom[2].data.T, bottom[1].data.T).T
        print 'A =', bottom[1].data.T
        
    def backward(self, top, propagate_down, bottom):
        #if propagate_down[0]:
        bottom[0].diff[...] = np.multiply(1-bottom[1].data.T, top[0].diff.T).T
        #if propagate_down[1]:
        bottom[1].diff[...] = np.reshape(np.sum(bottom[2].data-bottom[0].data, axis=1),(200,1))
        #print 'dA =', bottom[1].diff.T
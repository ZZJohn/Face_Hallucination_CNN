import caffe
import numpy as np

class MyTanHLayer(caffe.Layer):
    """A layer that just multiplies by ten"""

    def setup(self, bottom, top):
        pass

    def reshape(self, bottom, top):
        top[0].reshape(*bottom[0].data.shape)

    def forward(self, bottom, top):
        self.tanh = np.tanh(bottom[0].data[...])
        #print 'self.tanh =', self.tanh.T
        top[0].data[...] = 0.5*self.tanh+0.5
        #print 'top[0].data =', top[0].data.T

    def backward(self, top, propagate_down, bottom):
        bottom[0].diff[...] = 0.5*(1 - self.tanh*self.tanh) * top[0].diff
        #print bottom[0].diff[...]
        # diff * bottom is not fucking 1e-5 ??
        
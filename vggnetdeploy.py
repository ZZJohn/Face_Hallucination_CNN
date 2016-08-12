import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt

def im2col(im, fsize):
    C, H, W = im.shape
    col_extent = W - fsize + 1
    row_extent = H - fsize + 1
    start_idx = np.arange(fsize)[:, None] * W + np.arange(fsize)
    didx = H * W * np.arange(C)
    start_idx = (didx[:, None] + start_idx.ravel()).reshape((-1, fsize, fsize))
    offset_idx = np.arange(row_extent)[:, None] * W + np.arange(col_extent)
    out = np.take(im, start_idx.ravel()[:, None] + offset_idx[::1, ::1].ravel())
    return out

def conv_layer(input, fsize, filter, bias, numfilter, pad):
    C, H, W = input.shape
    input = np.lib.pad(input, (pad, pad), mode='constant', constant_values=0)
    input = np.delete(input, (0), axis=0)
    input = np.delete(input, (C), axis=0)
    col = im2col(input, fsize)
    col = col.T
    filter = filter.reshape(numfilter, C*fsize*fsize)
    filter = filter.T
    output = np.dot(col, filter)
    output = np.add(output, bias)
    output = output.T
    output = output.reshape(numfilter, H, W)
    return output

def maxpool_layer(input):
    C, H, W = input.shape
    input = input.reshape(C, H / 2, 2, W / 2, 2).swapaxes(2, 3).reshape(C, H / 2, W / 2, 4)
    return np.amax(input, axis=3)

def relu_layer(input):
    return input * (input > 0)

def ip_layer(input, filter, bias):
    if (len(input.shape) == 2):
        C = 1
        H, W = input.shape
    else:
        C, H, W = input.shape
    input = input.reshape(C*H*W, 1)
    output = np.dot(filter, input)
    output = output.T
    output = np.add(output, bias)
    return output

def dropout_layer(input, fraction):
    output = input * (1 - fraction)
    return output

weights = sio.loadmat('vgg_model.mat')
filter_conv11 = weights['conv1_1_w']
filter_conv12 = weights['conv1_2_w']
filter_conv21 = weights['conv2_1_w']
filter_conv22 = weights['conv2_2_w']
filter_conv31 = weights['conv3_1_w']
filter_conv32 = weights['conv3_2_w']
filter_conv33 = weights['conv3_3_w']
filter_conv41 = weights['conv4_1_w']
filter_conv42 = weights['conv4_2_w']
filter_conv43 = weights['conv4_3_w']
filter_conv51 = weights['conv5_1_w']
filter_conv52 = weights['conv5_2_w']
filter_conv53 = weights['conv5_3_w']
filter_ip1 = weights['fc6_w']
filter_ip2 = weights['fc7_w']
filter_ip3 = weights['fc8_w']
bias_conv11 = weights['conv1_1_b']
bias_conv12 = weights['conv1_2_b']
bias_conv21 = weights['conv2_1_b']
bias_conv22 = weights['conv2_2_b']
bias_conv31 = weights['conv3_1_b']
bias_conv32 = weights['conv3_2_b']
bias_conv33 = weights['conv3_3_b']
bias_conv41 = weights['conv4_1_b']
bias_conv42 = weights['conv4_2_b']
bias_conv43 = weights['conv4_3_b']
bias_conv51 = weights['conv5_1_b']
bias_conv52 = weights['conv5_2_b']
bias_conv53 = weights['conv5_3_b']
bias_ip1 = weights['fc6_b']
bias_ip2 = weights['fc7_b']
bias_ip3 = weights['fc8_b']


im = plt.imread('train_face/996.jpg')
im = np.array(im)
H, W, C = im.shape
data = np.zeros((C, H, W))
data[0, :, :] = im[:, :, 0]
data[1, :, :] = im[:, :, 1]
data[2, :, :] = im[:, :, 2]
avg = np.average(data, axis=(1,2))
std = np.std(data, axis=(1,2))
data[0, :, :] = (data[0, :, :] - avg[0]) / std[0]
data[1, :, :] = (data[1, :, :] - avg[1]) / std[1]
data[2, :, :] = (data[2, :, :] - avg[2]) / std[2]


conv1_1 = conv_layer(data, 3, filter_conv11, bias_conv11, 64, 1)
conv1_1 = relu_layer(conv1_1)
conv1_2 = conv_layer(conv1_1, 3, filter_conv12, bias_conv12, 64, 1)
conv1_2 = relu_layer(conv1_2)
pool1 = maxpool_layer(conv1_2)
conv2_1 = conv_layer(pool1, 3, filter_conv21, bias_conv21, 128, 1)
conv2_1 = relu_layer(conv2_1)
conv2_2 = conv_layer(conv2_1, 3, filter_conv22, bias_conv22, 128, 1)
conv2_2 = relu_layer(conv2_2)
pool2 = maxpool_layer(conv2_2)
conv3_1 = conv_layer(pool2, 3, filter_conv31, bias_conv31, 256, 1)
conv3_1 = relu_layer(conv3_1)
conv3_2 = conv_layer(conv3_1, 3, filter_conv32, bias_conv32, 256, 1)
conv3_2 = relu_layer(conv3_2)
conv3_3 = conv_layer(conv3_2, 3, filter_conv33, bias_conv33, 256, 1)
conv3_3 = relu_layer(conv3_3)
pool3 = maxpool_layer(conv3_3)
conv4_1 = conv_layer(pool3, 3, filter_conv41, bias_conv41, 512, 1)
conv4_1 = relu_layer(conv4_1)
conv4_2 = conv_layer(conv4_1, 3, filter_conv42, bias_conv42, 512, 1)
conv4_2 = relu_layer(conv4_2)
conv4_3 = conv_layer(conv4_2, 3, filter_conv43, bias_conv43, 512, 1)
conv4_3 = relu_layer(conv4_3)
pool4 = maxpool_layer(conv4_3)
conv5_1 = conv_layer(pool4, 3, filter_conv51, bias_conv51, 512, 1)
conv5_1 = relu_layer(conv5_1)
conv5_2 = conv_layer(conv5_1, 3, filter_conv52, bias_conv52, 512, 1)
conv5_2 = relu_layer(conv5_2)
conv5_3 = conv_layer(conv5_2, 3, filter_conv53, bias_conv53, 512, 1)
conv5_3 = relu_layer(conv5_3)
pool5 = maxpool_layer(conv5_3)
fc6 = ip_layer(pool5, filter_ip1, bias_ip1)
fc6 = relu_layer(fc6)
fc6 = dropout_layer(fc6, 0.5)
fc7 = ip_layer(fc6, filter_ip2, bias_ip2)
fc7 = relu_layer(fc7)
fc7 = dropout_layer(fc7, 0.5)
fc8 = ip_layer(fc7, filter_ip3, bias_ip3)
result = np.where(fc8 == np.max(fc8))
print(result)


'''
A = np.array([[[1, 2, 0], [1, 1, 3], [0, 2, 2]],
     [[0, 2, 1], [0, 3, 2], [1, 1, 0]],
     [[1, 2, 1], [0, 1, 3], [3, 3, 2]]])
filter = np.array([[1, 1, 2, 2, 1, 1, 1, 1, 0, 1, 1, 0], [1, 0, 0, 1, 2, 1, 2, 1, 1, 2, 2, 0]])
filter = filter.T
output = conv_layer(A, 2, filter, 2, 1)
output1 = maxpool_layer(output)
output2 = relu_layer(output1)
print(output)
print(output1)
print(output2)
'''
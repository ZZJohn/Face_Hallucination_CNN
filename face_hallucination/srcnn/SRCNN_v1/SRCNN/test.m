close all;
clear all;

load('x3.mat');
weights_conv1 = permute(weights_conv1, [2 1 3]);
weights_conv2 = permute(weights_conv2, [2 1 3]);
weights_conv3 = permute(weights_conv3, [2 1 3]);
weights_conv1 = reshape(weights_conv1, [9 9 3 64]);
weights_conv2 = reshape(weights_conv2, [1 1 64 32]);
weights_conv3 = reshape(weights_conv3, [5 5 32 3]);

caffe.set_mode_gpu();
caffe.set_device(0);
model = 'SRCNN_mat.prototxt';
net = caffe.Net(model, 'test');
net.params('conv1', 1).set_data(weights_conv1);
net.params('conv2', 1).set_data(weights_conv2);
net.params('conv3', 1).set_data(weights_conv3);
net.params('conv1', 2).set_data(biases_conv1);
net.params('conv2', 2).set_data(biases_conv2);
net.params('conv3', 2).set_data(biases_conv3);

res = [0,0];

for i=1:1000
    [psnr_bic, psnr_srcnn] = test_with_gpu(net,[num2str(i) '.jpg']);
    res = res + [psnr_bic, psnr_srcnn];
end

'PSNR:'
res = res * 0.001
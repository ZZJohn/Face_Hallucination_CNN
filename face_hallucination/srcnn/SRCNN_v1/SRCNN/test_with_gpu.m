function [psnr_bic, psnr_srcnn] = test_with_gpu(net,imfile)
    %% read ground truth image
    im  = imread(['train_face\' imfile]);

    %% set parameters
    up_scale = 224/64;

    %% work on illuminance only
    if size(im,3)==1
        im = repmat(im, [1 1 3]);
    end
    im_gnd = modcrop(im, up_scale);
    im_gnd = im2double(im_gnd);

    %% bicubic interpolation
    im_l = imresize(im_gnd, 1/up_scale, 'bicubic');
    im_b = imresize(im_l, up_scale, 'bicubic');
    [hei,wid,~] = size(im_b);
    im_cnn = im_b;

    %% SRCNN
%     size_input = 33;
%     size_label = 21;
%     stride = 12;
%     padding = abs(size_input - size_label)/2;
%     loss = [];
%     
%     for x = 1 : stride : hei-size_input+1
%         for y = 1 :stride : wid-size_input+1
%             
%             subim_input = im_b(x : x+size_input-1, y : y+size_input-1, :);
%             subim_label = im_gnd(x+padding : x+padding+size_label-1, y+padding : y+padding+size_label-1, :);
%             res = net.forward({subim_input, subim_label});
%             loss = [loss;res{1}];
%             subim_output = net.blobs('conv3').get_data();
%             im_cnn(x+padding : x+padding+size_label-1, y+padding : y+padding+size_label-1, :) = subim_output;
%             
%         end
%     end

    %% compute PSNR
    psnr_bic = psnr(im_gnd,im_b);
    psnr_srcnn=0;%psnr_srcnn = psnr(im_gnd,im_cnn);

    %% show results
    %mean_loss = mean(loss, 1)
    %fprintf('PSNR for Bicubic Interpolation: %f dB\n', psnr_bic);
    %fprintf('PSNR for SRCNN Reconstruction: %f dB\n', psnr_srcnn);

    %figure, imshow(im_b); title('Bicubic Interpolation');
    %figure, imshow(im_h); title('SRCNN Reconstruction');

    imwrite(im_b, ['bic\' imfile]);
    imwrite(im_cnn, ['srcnn\' imfile]);
end
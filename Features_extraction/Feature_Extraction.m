%% The code of this script aims to extract features from the AlexNet
%  neural network 

clear all
close all

% *****************************************************
% ************ Set CNN as Extractor *******************
% *****************************************************
run  matlab/vl_setupnn

net = load('imagenet-caffe-alex.mat') ;
net2 = net;
net2.layers = net2.layers(1:2);
dimension = net.meta.normalization.imageSize(1:2);
imm_avg = net.meta.normalization.averageImage;

% *****************************************************
% ************ Data structures      *******************
% *****************************************************
feature_collection = {};
imdir_m1 = 'BenMal';
fnames_m1 = dir(fullfile(imdir_m1, '*.png'));
nfiles_m1 = size(fnames_m1,1);
filenames_m1 = {};
for i = 1:nfiles_m1
    filenames_m1{i} = fnames_m1(i).name;
end

% *****************************************************
% ************ Feature extraction   *******************
% *****************************************************
nfiles_m1 = length(1:nfiles_m1);
tic
for f =1:nfiles_m1
    f
    file = strcat(imdir_m1,'/',filenames_m1{f});
    frame_m = imread(file);
    % EXTRACT ALEXNET FEATURES
    im_ = single(frame_m) ; % note: 255 range
    im_ = imresize(im_, dimension) ;
    im_ = im_ - imm_avg;
    % run the CNN up to conv3
    res = vl_simplenn(net2, im_) ;
    % get feature
    CNNfeature_m = res(end).x; % conv3 
    feature_collection{f} = CNNfeature_m;
end
toc

% *****************************************************
% ************ Save                 *******************
% *****************************************************
save save.mat feature_collection
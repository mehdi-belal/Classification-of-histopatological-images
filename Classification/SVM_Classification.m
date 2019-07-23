%% The code of this script is aimed at training a Support Vector Machine
%  based on the resized data extracted from CNN


clear all
close all

% *****************************************************
% ************ Loading Training Set *******************
% *****************************************************

disp('Loading Trainig Set');
load('Features Training Set\imagenet-caffe-alex_20_40X_Benigno_correlation_0.6.mat')
B = descriptors;
load('Features Training Set\imagenet-caffe-alex_20_40X_Maligno_correlation_0.6.mat')
M = descriptors;
data = [B,M];
D = size(data(1).descriptor,2);
nM = size(M,2);
nB = size(B,2);
data_descriptors_train = reshape([data.descriptor],nM+nB,D);
data_labels_train = [zeros(nB,1);ones(nM,1)];

disp('Loading Test Set');
load('Test Features\imagenet-caffe-alex_20_40X_Benigno_correlation_0.6_TestSet.mat')
B_T = descriptors;
load('Test Features\imagenet-caffe-alex_20_40X_Maligno_correlation_0.6_TestSet.mat')
M_T = descriptors;

data_test = [B_T,M_T];
D = size(data_test(1).descriptor,2);
nMT = size(M_T,2);
nBT = size(B_T,2);
data_descriptors_test = reshape([data_test.descriptor],nMT+nBT,D);
data_labels_test = [zeros(nBT,1);ones(nMT,1)];
data_names_test = [];


% *****************************************************
% ******* Preparing Data and Labels     ***************
% *****************************************************

for i = 1:nMT+nBT
    p = strsplit(data_test(i).image_name, '-');
    tmp = strrep(p(3), 'AB', '');
    tmp = strrep(tmp, 'B', '');
    tmp = strrep(tmp, 'CD', '');
    tmp = strrep(tmp, 'C', '');
    tmp = strrep(tmp, 'G', '');
    tmp = strrep(tmp, 'E', '');
    tmp = strrep(tmp, 'DE', '');
    tmp = strrep(tmp, 'D', '');
    tmp = strrep(tmp, 'EF', '');
    tmp = strrep(tmp, 'F', '');
    tmp = strjoin(tmp);
    data_names_test = [data_names_test; str2num(tmp)];
end

shuffle_vector_train = randperm(nM+nB);
shuffled_data_train = data_descriptors_train(shuffle_vector_train,:);
shuffled_labels_train = data_labels_train(shuffle_vector_train);


% *****************************************************
% ************ Training                  **************
% *****************************************************


disp('Training model SVM');
data_train_set = shuffled_data_train;
label_train_set = shuffled_labels_train;

data_test_set = data_descriptors_test;
label_test_set = data_labels_test;
names_test_set = data_names_test;

SVMModel = fitcsvm(data_train_set,label_train_set, 'Standardize',true);


% -------------------------------------
% SVM Optimization using RBF Kernel, used training set
% composed by 10 elements (K-Fold)

SVMModel = fitcsvm(data_train_set,label_train_set,'KernelFunction','rbf');
c = cvpartition(size(label_train_set,1),'KFold',10);

minfn = @(z)kfoldLoss(fitcsvm(data_train_set,label_train_set,'CVPartition',c,...
'KernelFunction','rbf','BoxConstraint',exp(z(2)),...
    'KernelScale',exp(z(1))));
opts = optimset('TolX',5e-4,'TolFun',5e-4);
disp('Tuning parametres')

%numero di iterazioni
m = 15;
fval = zeros(m,1);
z = zeros(m,2);
for j = 1:m;
    j
    [searchmin fval(j)] = fminsearch(minfn,randn(2,1),opts);
    z(j,:) = exp(searchmin);
end
z = z(fval == min(fval),:)

% Training with best parameters
SVMModel = fitcsvm(data_train_set,label_train_set,'KernelFunction','rbf',...
    'KernelScale',z(1),'BoxConstraint',z(2));

% Prediction
[label, score] = predict(SVMModel,data_test_set);



% *****************************************************
% ************ Results               ******************
% *****************************************************

disp('Loading results');
corretto = 0;
for i = 1:size(label_test_set,1)
    if label(i) == label_test_set(i)
        corretto = corretto + 1;
    end
end

% this part of the code is used to reorder the data according to the 
% labels, so as to be able to do the evaluation with the metric based 
% on the patients

names_test_set_sorted = sort(names_test_set);

tmp1 = [];
for i = 2:size(names_test_set_sorted,1)
    if names_test_set_sorted(i-1) ~=  names_test_set_sorted(i)
        tmp1 = [tmp1;names_test_set_sorted(i)];
    end
end

y = zeros(size(names_test_set_sorted));
for i = 1:length(names_test_set_sorted)
    y(i) = sum(names_test_set_sorted==names_test_set_sorted(i));
end

names_test_set_sorted = unique(names_test_set_sorted);

number_images = [];

for i = 1:length(names_test_set_sorted)
    tmp = 0;
    for j = 1:length(names_test_set)
        if names_test_set_sorted(i) == names_test_set(j)
            tmp = tmp+1;
        end
    end
    number_images(i) = tmp;
end

result = [];


% evaluation of the results according to the patient level metric, the 
% global recognition rate is based on the images correctly classified for 
% each patient

for i=1:length(names_test_set_sorted)
    tmp = 0;
    for j=1:length(names_test_set)
        if names_test_set_sorted(i) == names_test_set(j)
             if label(j) == label_test_set(j)
                 tmp = tmp+1;
                 ends
        end
    end
    result(i)=(tmp)/(number_images(i));
end

disp('results average');
mean(result)
disp('results standard deviation');
std(result)

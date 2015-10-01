%% Execute once during the training
%addpath('../matlab');
%addpath('../SLIC');
%vl_compilenn;

%% Define some variables
global net;

path = '../data/';
files = dir([path '*.jpg']);
grounTs = dir([path '*.png']);

disp('Loading the Model ...')
net = load('../models/imagenet-vgg-verydeep-16.mat');
clc;
%% Start the procedure and repeat for all images
create = true;
curr_pos = 0;

for i=1:10
    fprintf('Processing %d/10...\n',i);
    
    im = imread([path files(i).name]);
    gt = imread([path grounTs(i).name]);
    
    input{1} = im;
    input{2} = flip(im,2);
    
    
    tic;
    
    %[fmax,fmaxCR,fmaxBG,fmaxEI,overlap,label] = featureComputation(input,gt,net);
    [data,label] = featureComputation(input,gt,net);
    %load('data');  
    
    curr_pos = store2hdf5('trial.h5',data,label,create,1000,curr_pos);    
    create = false;
    
    fprintf('Elapsed Time: %f\n',toc);
    disp('-----------------------------------------------')
end
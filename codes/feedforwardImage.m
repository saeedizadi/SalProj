function res =  feedforwardImage(im,net)
%vl_compilenn('enableGpu',true,'cudaMethod','nvcc');
%net = load('models/vgg/imagenet-vgg-verydeep-16');
% obtain and preprocess an image
%im = imread('../data/train/car/car_100027.png');
im_ = single(im) ;
im_ = imresize(im_, net.normalization.imageSize(1:2)) ;
im_ = im_ - net.normalization.averageImage ;
%n the CNN
res = vl_simplenn(net, im_) ;

% show the classification result
% scores = squeeze(gather(res(end).x)) ;
% [bestScore, best] = max(scores) ;
% figure(1) ; clf ; imagesc(im) ;
% title(sprintf('%s (%d), score %.3f',...
% net.classes.description{best}, best, bestScore)) ;    
end                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            
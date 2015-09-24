%function [featureMax,featureMaxCR,featureMaxBG,featureMaxEI,overlap,label] = featureComputation(input,gt,net)
function [data,label] = featureComputation(input,gt,net)

layerInx = [3 5 8 10 13 15 17 20 22 24 27 29 31];
mirFlag = false;

bgMask = zeros(256); bgMask(1:15,:) = 1;bgMask(:,1:15) = 1;bgMask(end-15:end,:) = 1;bgMask(:,end-15:end) = 1;
bgMaskVec = logical(bgMask(:));

for nInput=1:length(input)
    
    %Load each image in the directory and resize it into 256,256
    im = input{nInput};
    im = imresize(im,[256,256]);
    gt = logical(imresize(gt,[256,256]));
    
    
    %Generating Superpixels for the image
    disp('Generating Superpixels ...')
    
    if mirFlag == false
        [l, Am, ~, ~] = slic(im, 500, 60,3 ,'mean');
    else
        l = flip(l,2);
    end
    
    numOfSPs = numel(unique(l));
    
    
    %Feedforward the input image and the intermediate features
    disp('Feeding the Input Image Forward through the Net ...')
    res = feedforwardImage(im,net);
    
    %Extracting the Inermediate Features and resizing them into [256,256]
    features = cell(1,13);
    
    for k = 1:numel(layerInx)
        temp1 = imresize(res(layerInx(k)).x,[256,256]);
        features{k} = double(temp1);
    end
    
    %Concatenating the feature maps -> [256,256,4224]    ]
    features = cat(3,features{:});
    
    %Vectorzing Maps -> [65536 4224]
    [~,~,depth] = size(features);
    featureVec = reshape(features,[65536,depth]);
    
    
    %Perfor Avg. Pooling across every superpixel
    fprintf('\nProcessing superpixels...\n');
    for spNum = 1: numOfSPs
        %fprintf('\b\b\b\b\ %3d ',spNum)
        
        %Compute mask for superpixels and context region
        
        %Use neighboring superpixels as the context area instead of
        %rectangular area
        [idxVec,masks] = neighSPs(l,Am,spNum,3);
        
        %mask = ismember(l,spNum);
        %idxVec = contextRegion(l,mask,1);
        
        %Compute the overlap and label for the superpixel
        [overlap(spNum),label(spNum)] = computeOverlap(masks{1},gt);
        
        %Vectorize mask for superpixel
        maskVec = masks{1}(:);
        
        
        featureSPAvg = mean(featureVec(maskVec,:),1);
        featureCRAvg = mean(featureVec(idxVec,:),1);
        
        featuresSP(spNum,:) = featureSPAvg;
        featuresCR(spNum,:) = featureCRAvg;
        
        %featuresSP(spNum,:) = featureSPAvg/norm(featureSPAvg,2);
        %featuresCR(spNum,:) = featureCRAvg/norm(featureCRAvg,2);
        
    end
    
    if(mirFlag == false)
        featureOrg = featuresSP;
        featureOrgCR = featuresCR;
        featureOrgBG = mean(featureVec(bgMaskVec,:),1);
        featureOrgEI = mean(featureVec,1);
        
        clearvars -except input net nInput mirFlag layerInx l featureOrg featureOrgCR  featureOrgBG featureOrgEI bgMaskVec overlap label gt Am
    else
        featureMir = featuresSP;
        featureMirCR = featuresCR;
        featureMirBG = mean(featureVec(bgMaskVec,:),1);
        featureMirEI = mean(featureVec,1);
    end
    mirFlag = true;
    
    if(nInput == 2)
        featureMax = max(featureOrg,featureMir);
        featureMaxCR = max(featureOrgCR,featureMirCR);
        featureMaxBG = max(featureOrgBG,featureMirBG);
        featureMaxEI = max(featureOrgEI,featureMirEI);
        
        featureLocContrast = gsqrt((featureMax-featureMaxCR).^2);
        featureBgContrast = gsqrt(featureMax-repmat(featureMaxBG,[size(featureMax,1) 1]).^2);        
        
        clearvars -except featureMax featureMaxCR featureLocContrast featureBgContrast label overlap
        
        data = [featureMax featureMaxCR featureLocContrast featureBgContrast];
        clearvars -except data label overlap
    end
    
end
end


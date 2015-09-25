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
    
    if mirFlag == false
        fprintf('\nWorking with Superpixels (Original) ...\n')
        [l, Am, ~, ~] = slic(im, 500, 60,3 ,'mean');
    else
        fprintf('Working with Superpixels (Mirror) ...\n')
        l = flip(l,2);
        gt = flip(gt,2);
    end
    
    numOfSPs = numel(unique(l));
    
    
    %Feedforward the input image and the intermediate features
    %disp('Feeding the Input Image Forward through the Net ...')
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
    count = 0;
    for spNum = 1: numOfSPs                
        
        %Use neighboring superpixels as the context area instead of
        %rectangular area
        [idxVec,masks] = neighSPs(l,Am,spNum,3);
        
        %Vectorize mask for superpixel
        maskVec = masks{1}(:);
        
        %Compute the overlap and label for the superpixel
        if mirFlag == false
            [~ , tempLabel] = computeOverlap(masks{1},gt);
        else            
            [~ , tempLabel] = computeOverlap(masks{1},gt);
        end
        if ~isequal(tempLabel,-1)
            count = count+1;
            
            label(count) = tempLabel;
            
            featuresSP(count,:) = mean(featureVec(maskVec,:),1);
            featuresCR(count,:) = mean(featureVec(idxVec,:),1);
                                     
            %featuresSP(spNum,:) = featureSPAvg/norm(featureSPAvg,2);
            %featuresCR(spNum,:) = featureCRAvg/norm(featureCRAvg,2);
        end
        
    end
    
    if(mirFlag == false)
        featureOrg = featuresSP;
        featureOrgCR = featuresCR;
        featureOrgBG = mean(featureVec(bgMaskVec,:),1);
        
        clearvars -except overlap input net nInput mirFlag layerInx l featureOrg featureOrgCR  featureOrgBG bgMaskVec overlap label gt Am
    else
        featureMir = featuresSP;
        featureMirCR = featuresCR;
        featureMirBG = mean(featureVec(bgMaskVec,:),1);
    end
    mirFlag = true;
    
    if(nInput == 2)
        featureMax = max(featureOrg,featureMir);
        featureMaxCR = max(featureOrgCR,featureMirCR);
        featureMaxBG = max(featureOrgBG,featureMirBG);
        
        featureLocContrast = gsqrt((featureMax-featureMaxCR).^2);
        featureBgContrast = gsqrt(featureMax-repmat(featureMaxBG,[size(featureMax,1) 1]).^2);
        
        clearvars -except featureMax featureMaxCR featureLocContrast featureBgContrast label overlap
        
        data = [featureMax featureMaxCR featureLocContrast featureBgContrast];
        clearvars -except data label overlap
    end
    
end
end


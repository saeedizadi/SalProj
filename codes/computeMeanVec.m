function [meanVec,stdVec] = computeMeanVec(numOfInst,numOfDim)
   meanVec = zeros(1,numOfDim);
   stdVec = zeros(1,numOfDim);
   
   %textprogressbar('Calculating mean and std of features: ');   
   for i=1:numOfDim       
       %textprogressbar((i/numOfDim)*100);
       
       %Retreive the desired data
       tic;
       dataTemp = h5read('/media/saeed/88AAF556AAF5416C/trial.h5','/data',[1 i 1 1],[1,1,1,numOfInst]);
       %Reshape it from 4-D into row-dim
       dataTemp = reshape(dataTemp,[1 numOfInst]);
       
       %Compute mean and std of the reshaped vecor
       meanVec(i) = mean(dataTemp);
       stdVec(i) = std(dataTemp);
       clear dataTemp
       fprintf('Elapsed Time for dim %d : %f\n',i,toc);
   end
textprogressbar('Done');
end

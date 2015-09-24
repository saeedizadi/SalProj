function [overlap,label] = computeOverlap(maskSP, gt)

%Check the overlap with foreground region
temp = gt .* maskSP;
fgOverlap = sum(sum(temp))/sum(sum(maskSP));

%Check the overlap with background region
temp = ~gt .* maskSP;
bgOverlap = sum(sum(temp))/sum(sum(maskSP));

%if the sp is 80% on the foreground, it is salient else background
%if the sp is neither 80% foreground nor background, then reject it
if fgOverlap > 0.8 && bgOverlap <0.2
    overlap = fgOverlap;
    label = 1;
elseif fgOverlap < 0.2 && bgOverlap > 0.8
    overlap = bgOverlap;
    label = 0;
else
    label = -1;
    overlap = 0;
end

end 
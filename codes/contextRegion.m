function [idx] = contextRegion(l,supMask,alpha)

[x,y] = find(supMask == 1);
%[x,y] = find(l ==supMask);

%mask = zeros(size(l));
%mask(sub2ind(size(l),y,x)) = 1;

minx = min(x);maxx = max(x);miny = min(y);maxy = max(y);

w = floor(alpha*(maxy-miny+1));
h = floor(alpha*(maxx-minx+1));

xc = floor((minx+maxx)/2);
yc = floor((miny+maxy)/2);

if xc>size(l,1)
    xc = size(l,1);
end
if yc>size(l,2)
    yc = size(l,2);
end

X = [-h/2 h/2 ];
Y = [-w/2 w/2 ];

P = [X;Y];

P(1,:) = P(1,:)+yc;
P(2,:) = P(2,:)+xc;

%figure;imshow(img);
%hold on
%plot(P(1,:),P(2,:),'r-');
%contextPatch = imcrop(img,[min(P(1,:)) min(P(2,:)) w h]);
ConRegMask = zeros(256);
a = round(max(1,min(P(2,:))));
b = round(min(256,max(1,min(P(2,:)))+h));
c = round(max(1,min(P(1,:))));
d = round(min(256,max(1,min(P(1,:))) + w));
ConRegMask(a:b,c:d) = 1;
%temppp = reshape(ConRegMask,[65536 1]);
temppp = ConRegMask(:);
idx = find(temppp == 1);
%ConRegMask = imresize(ConRegMask,[256,256]);

figure;imshow(ConRegMask)

end


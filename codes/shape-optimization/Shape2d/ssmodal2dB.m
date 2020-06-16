function [B]=ssmodal2dB(I,n,c,nodesa,dxa,M)
B=zeros(n,1);


for i=1:n
    B(i,1)=dxa^2*sum(basesin(nodesa(:,2),nodesa(:,3),I(i,2),I(i,3)).*nodesa(:,4));
end
B=M\B;
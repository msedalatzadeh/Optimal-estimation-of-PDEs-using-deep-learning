function [B]=Bnew(N,nodesa,dxa,Bo)
global nodes

B1=zeros(N,1);
for ii=1:N
    B1(ii,1)=dxa*sum(nodesa.*sin(ii*pi*nodes'));
end
B=[0*B1;Bo*B1];
function [A,Bo,Q]=EBssfd(N)
EI=.1;



%Mass
M=eye(N);
[~,~,K1]=laplacian(N,{'DD'});
K=(EI*N^2)*(K1*K1);
%K=(EI*N)*(K1);
%d=@(x) 0+x.*0;
d=1e-3;
D=zeros(N,N);
%D=d*K;
D=eye(N);
a1=M\K;
a2=M\D;
b1=M\eye(N);


A=[0*M eye(N);-a1 -a2];
Bo=b1;
Q=[M 0*M;0*M M];
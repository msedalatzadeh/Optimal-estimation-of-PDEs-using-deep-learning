function [A,Bo,Q,KK]=EBss(N)
EI=.1;
A=zeros(N);


%Mass
M=zeros(N,N);
for i=1:N
        M(i,i)=integral(@(x) sin(i*pi*x).^2,0,1);
end

K=zeros(N,N);
for i=1:N
        K(i,i)=integral(@(x) EI*(i*pi)^4*sin(i*pi*x).^2,0,1);
end

K1=zeros(N,N);
for i=1:N
        K1(i,i)=integral(@(x) EI*(i*pi)^2*sin(i*pi*x).^2,0,1);
end

dv=@(x)1e-3+x.*0;

DV=zeros(N,N);

for j=1:N
    for i=1:N
        DV(i,j)=integral(@(x) dv(x).*sin(i*pi*x).*sin(j*pi*x),0,1);
    end
end

dkv=1e-4;
DKV=dkv*K;


A=[0*M eye(N);-inv(M)*K -inv(M)*(DV+DKV)];
Bo=inv(M);
Q=[M+K1 0*M;0*M M];
KK=[K1 0*K1;0*K1 K1];

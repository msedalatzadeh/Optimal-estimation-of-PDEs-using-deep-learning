function [U,XX,VV,T,costnew]=ocsimu(control,N,x0,nodesa,gamma,gammaw)
global A Bo Q dxa nodes KK budget
[B]=Bnew(N,nodesa,dxa,Bo);
R=gamma;
Tf=10;
dt=0.005;
[Kr,Pr,~ ]=lqr(A,B,Q,R);
T=0:dt:Tf;
xm=x0;

U=[];
if (control==1)
cl=A-B*Kr;
else
cl=A;
end
    
icl=(eye(2*N)-0.5*dt*cl)\(eye(2*N)+0.5*dt*cl);
timesi=0;
% [maxv,maxe]=eig(Pr,Q);
% xm=maxv(:,end);
% xm=xm/sqrt(xm'*Q*xm);
X=xm;
costnew=xm'*Pr*xm+0.5*gammaw*(dxa*sum(nodesa)-budget)^2;
%plot(xm)
Adj=[];

while (timesi<Tf)
    Xc=icl*X(:,end);
    X=[X Xc];
    timesi=timesi+dt;
end
Adj=Pr*X;
U=-R^(-1)*B'*Adj;

XX=zeros(length(nodes),size(X,2));
for ii=1:N
    XX=XX+sin(pi*nodes'*ii)*X(ii,:);
end
VV=zeros(length(nodes),size(X,2));
for ii=1:N
    VV=VV+sin(pi*nodes'*ii)*X(N+ii,:);
end


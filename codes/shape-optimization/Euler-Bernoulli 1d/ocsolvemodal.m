function [gtot,ngtot,costnew]=ocsolvemodal(N,x0,nodesa,gamma,gammaw)
global A Bo Q dxa nodes KK budget
[B]=Bnew(N,nodesa,dxa,Bo);
R=gamma;
Tf=200;
dt=0.025;
[Kr,Pr,~ ]=lqr(A,B,Q,R);

T=0:dt:Tf;


xm=x0;
% [maxv,maxe]=eig(Pr,Q);
% xm=maxv(:,end);
% xm=xm/sqrt(xm'*Q*xm);

costnew=xm'*Pr*xm+0.5*gammaw*(dxa*sum(nodesa)-budget)^2;
U=[];
Adj=[];
cont=1;
Loc=[];
g=[];


cl=A-B*Kr;
icl=(eye(2*N)-0.5*dt*cl)\(eye(2*N)+0.5*dt*cl);
timesi=0;
X=xm;

while (timesi<Tf)
    Xc=icl*X(:,end);
    X=[X Xc];
    timesi=timesi+dt;
end


%[~,X]=ode45(@(t,x) cl*x,T,xm);
%X=X';
Adj=Pr*X;
U=-R^(-1)*B'*Adj;
g=Adj.*repmat(U,size(Adj,1),1);

% figure(666)
% mesh(X)



gtot=sum(g(1:N,:),2)*dt;

gtott=0*nodes;
for ii=1:N
    gtott=gtott+gtot(ii)*sin(pi*nodes*ii);
end
gtot=gtott'+gammaw*(dxa*sum(nodesa)-budget);
    
%ngtot=sqrt(gtot'*M*gtot);
ngtot=sqrt(gtot'*gtot);
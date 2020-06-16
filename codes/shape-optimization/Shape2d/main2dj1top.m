clear all
close all
clc



N=40; %Number of FEM
dx=1/N;
[x,y]=ndgrid(dx:dx:1-dx);

h=1/N^2;
ni=N-1;
nodesa=ones(ni^2,1);




c=1;  %wave parameter

tol=1e-5;
diff=1;
gamma=1e-5;
threshold=0.1;
alpha0=.01;
%phi0=-0.1+0*nodes';
phi0=(-1*nodesa+(1-nodesa));
phi0=.1*phi0;
costoldest=10000;
costold=1000;
costnew=0;
gtot=nodesa;
costev=[];
%gammaw=0.05;
gammaw=1e-2;
while(norm(alpha0)>=tol)

[gtot,ngtot,costnew]=ocsolvenew2d(c,N,nodesa,gamma,gammaw);
costev=[costev costnew];

phi1=(1-alpha0)*phi0+alpha0*gtot/ngtot;




nodesaux=(phi1<0).*1.0;
[~,~,costnewaux]=ocsolvenew2d(c,N,nodesaux,gamma,gammaw);
costnew
costnewaux

if ((costnewaux-costnew)<=0*1e-4)
    
costoldest=costold;
costold=costnew
diff=costold-costnew;
nodesold=nodesa;
nodesa=nodesaux;
phi0=phi1;
%alpha0=.01;



figure(1)
subplot(2,2,1)
surf(x,y,reshape(nodesold,ni,ni))
subplot(2,2,2)
surf(x,y,reshape(nodesa,ni,ni))
subplot(2,2,3)
surf(x,y,reshape(gtot+phi1,ni,ni))
subplot(2,2,4)
plot(costev)
drawnow


 else
     alpha0=0.9*alpha0
     costoldest=10000;
     diff=1;
 end




end
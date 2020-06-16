clear all
close all
clc

global A Bo Q dxa nodes

N=199; %Number of FEM
[A,Bo,Q]=EBssfd(N);

x0=0;  %domain start
x1=1;  %domain end
h=(x1-x0)/(N+1);
nodes=x0+h:h:x1-h;
nodesa=0*nodes;
dxa=h;
xa=0.5;
da=90*h;
inc=xa-da;
finc=xa+da;
[~,indinc]=min(abs(nodes-inc));
[~,indfinc]=min(abs(nodes-finc));
nodesa(indinc:indfinc)=1;nodesa=1*nodesa';
%load('nodesa.mat')
%nodesa(40:60)=0;
%nodesa=1-nodesa;
x0=zeros(2*N,1);
%x0(1,1)=1;
xinit=abs(1*sin(2*pi*nodes))+0;
x0=[xinit';0*xinit'];
%xinit=x.*(x-1)+sin(pi*x);
%xinit=sign(x-0.5);
%xinit=100*abs(x-0.7).^4.*x.*(x-1);
%xinit=max(sin(6*pi*x),0);


tol=1e-5;
diff=1;
gamma=1e-1;
threshold=0.1;
alpha0=.9;
%phi0=-0.1+0*nodes';
phi0=(-1*nodesa+(1-nodesa));
phi0=.1*lapreg(1e-4,phi0);




costoldest=10000;
costold=1000;
costnew=0;
gtot=nodesa;
costev=[];
%gammaw=0.05;
gammaw=1e3;
while(norm(alpha0)>=tol)

[gtot,ngtot,costnew]=ocsolvenew(N,x0,nodesa,gamma,gammaw);
costev=[costev costnew];


%level set update
gtot=lapreg(1e-4,gtot);
phi1=(1-alpha0)*phi0+alpha0*gtot/ngtot;




nodesaux=(phi1<0).*1.0;
[~,~,costnewaux]=ocsolvenew(N,x0,nodesaux,gamma,gammaw);
costnew
costnewaux

if ((costnewaux-costnew)<=1e-1)
    
costoldest=costold;
costold=costnew
diff=costold-costnew;
nodesold=nodesa;
nodesa=nodesaux;
phi0=phi1;
alpha0=.1;



figure(1)
subplot(1,3,1)
plot(nodes,nodesold,nodes,nodesa)
%subplot(1,3,2)
%plot(nodes,phi1,nodes,gtot)

subplot(1,3,2)
plot(nodes,xinit)
subplot(1,3,3)
plot(costev)
drawnow


 else
     alpha0=0.5*alpha0
     costoldest=10000;
     diff=1;
 end




end
save('nodesa.mat','nodesa')


clear all
close all
clc

global A M K y0p vol
n1=8;
n=n1^2;
[nx,ny]=ndgrid(1:n1);
nx=nx(:);
ny=ny(:);
c=.01;
gammau=1e-3;
gammaw=1e5;


for i=1:n
    I(i,:)=[i nx(i) ny(i)];
end



dxa=0.005;
[nxaa,nyaa]=ndgrid(dxa:dxa:1-dxa);
nxa=nxaa(:);
nya=nyaa(:);
na=length(nxa);
nia=sqrt(na);
vol=0.05;

nodesa(:,1)=1:na;
nodesa(:,2)=nxa;
nodesa(:,3)=nya;
nodesa(:,4)=((abs(nxa-0.5).^2+abs(nya-0.5).^2)<1.5*vol/pi);
%nodesa(:,4)=(nxa>0.1).*(nxa<0.3).*(nya>0.3).*(nya<0.7)+(nxa>0.7).*(nxa<0.9).*(nya>0.3).*(nya<0.7);
%nodesa(:,4)=1;


%% %%
y0=@(x,y) (1-x).*x.*y.*(1-y)/0.05;
%  %y0=@(x,y) sin(pi*x).*sin(pi.*y);
% %  y0=@(x,y) (x>0.5).*(y>0.5);
y0p=zeros(n,1);
for i=1:n
     y0p(i,1)=integral2(@(x,y) y0(x,y).*basesin(x,y,I(i,2),I(i,3)),0,1,0,1);
 end
%% 



tol=1e-6;
diff=1;
%phi0=.01*phi0;

alpha0=.01;
phi0=nodesa(:,4)-0.1;
phi0=-.01*sign(phi0);
costoldest=10000;
costold=1000;
costnew=0;
gtot=nodesa(:,4);
costev=[];

[A,B,M,K]=ssmodal2d(I,n,c,nodesa,dxa);
it=1;
thetaev=[];

while(norm(alpha0)>=tol)


[gtot,ngtot,costnew]=ocsolve2dmodal(c,n,I,dxa,na,nodesa,gammau,gammaw,costold);
costev=[costev costnew];
theta=acos(phi0'*gtot*dxa^2/ngtot);
phi1=(sin((1-alpha0)*theta)*phi0+sin(alpha0*theta)*gtot/ngtot)/sin(theta);



nodesaux=nodesa;
nodesaux(:,4)=(phi1<0).*1.0;
[~,~,costnewaux]=ocsolve2dmodal(c,n,I,dxa,na,nodesaux,gammau,gammaw,costold);
costnew
costnewaux

if ((costnewaux-costnew)<=0*1e-4)
    
costoldest=costold;
costold=costnew
diff=costold-costnew;
nodesold=nodesa;
nodesa=nodesaux;
phi0=phi1;
alpha0=min(0.01,10*alpha0);
thetaev=[thetaev theta];

% 
% if(mod(it,10)==0)
%     it
% phi0=nodesold(:,4)-0.1;
% phi0=-.01*sign(phi0);
% end
% it=it+1
% 


figure(1)
subplot(2,3,1)
mesh(nxaa,nyaa,reshape(nodesa(:,4),nia,nia))
subplot(2,3,2)
mesh(nxaa,nyaa,reshape(gtot,nia,nia))
subplot(2,3,3)
mesh(nxaa,nyaa,reshape(phi0,nia,nia))
subplot(2,3,4)
plot(costev)
subplot(2,3,5)
plot(thetaev)
drawnow


 else
     alpha0=0.9*alpha0
     costoldest=10000;
     diff=1;
 end




end

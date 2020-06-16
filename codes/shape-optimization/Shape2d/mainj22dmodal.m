

clear all
close all
clc

global A M K y0p vol
n1=10;
n=n1^2;
[nx,ny]=ndgrid(1:n1);
nx=nx(:);
ny=ny(:);
c=.01;
gammau=1e-3;
gammaw=1e1;


for i=1:n
    I(i,:)=[i nx(i) ny(i)];
end

vol=0.04;


dxa=0.005;
[nxaa,nyaa]=ndgrid(dxa:dxa:1-dxa);
nxa=nxaa(:);
nya=nyaa(:);
na=length(nxa);
nia=sqrt(na);

% 
%load('nodesold.mat');
%nodesa=nodesold;
% 
 nodesa(:,1)=1:na;
 nodesa(:,2)=nxa;
 nodesa(:,3)=nya;
 nodesa(:,4)=((abs(nxa-0.5).^2+abs(nya-0.5).^2)<2*vol/pi);

% % % 

%nodesa(:,4)=((abs(nxa-0.5).^2+abs(nya-0.5).^2)<1.5*vol/pi).*nodesa(:,4);
% % % %nodesa(:,4)=(nxa>=0.5);
%nodesa(:,4)=1;

%% %%
%y0=@(x,y) (1-x).*x.*y.*(1-y)/0.05;
%y0=@(x,y) sin(pi*x).*sin(pi.*y);
% %  y0=@(x,y) (x>0.5).*(y>0.5);
%y0=@(x,y) max(sin(3*pi*x),0).^3.*sin(pi*y).^3;
y0=@(x,y) max(sin(4*pi*(x-1/8)),0).^3.*sin(pi*y).^3;
y0p=zeros(n,1);
for i=1:n
     y0p(i,1)=integral2(@(x,y) y0(x,y).*basesin(x,y,I(i,2),I(i,3)),0,1,0,1);
 end
%% 



tol=1e-6;
diff=1;

alpha0=.01;
%phi0=(-1*nodesa(:,4)+(1-nodesa(:,4)));
phi0=levelsetreset(dxa,reshape(nodesa(:,4),199,199));
phi0=.01*phi0(:);
costoldest=10000;
costold=1000;
costnew=0;
gtot=nodesa(:,4);
costev=[];

[A,B,M,K]=ssmodal2d(I,n,c,nodesa,dxa);

it=0;

while(norm(alpha0)>=tol)
    
it=it+1
%level set update
% if (mod(it,100)==0)
% % %phi0=lapreg(1e-3,-nodesold);
% % %fprintf('entering the super duper Kalise eikonal')
% phi0=levelsetreset(dxa,reshape(nodesold(:,4),199,199));
% phi0=alpha0*max(abs(gtot/ngtot))*phi0(:);
% % gammaw=1.3*gammaw;
% end

[gtot,ngtot,costnew]=ocsolve2dmodal(c,n,I,dxa,na,nodesa,gammau,gammaw,costold);
costev=[costev costnew];
phi1=(1-alpha0)*phi0+alpha0*gtot/ngtot;



nodesaux=nodesa;
nodesaux(:,4)=(phi1<0).*1.0;
[~,~,costnewaux]=ocsolve2dmodal(c,n,I,dxa,na,nodesaux,gammau,gammaw,costnew);
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



figure(1)
subplot(2,2,1)
mesh(nxaa,nyaa,reshape(nodesa(:,4),nia,nia))
subplot(2,2,2)
mesh(nxaa,nyaa,reshape(gtot,nia,nia))
subplot(2,2,3)
mesh(nxaa,nyaa,reshape(phi1,nia,nia))
subplot(2,2,4)
plot(costev)
drawnow


 else
     alpha0=0.9*alpha0
     costoldest=10000;
     diff=1;
 end




end

%save('nodesold.mat','nodesold');

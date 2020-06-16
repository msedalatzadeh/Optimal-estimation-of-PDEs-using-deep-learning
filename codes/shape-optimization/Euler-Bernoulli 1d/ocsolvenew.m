function [gtot,ngtot,costnew]=ocsolvenew(N,x0,nodesa,gamma,gammaw)
global A Bo Q dxa nodes
[B]=Bnewfd(N,nodesa,dxa,Bo);
R=gamma;
Tf=100;
dt=0.1;
[Kr,Pr,~ ]=lqr(A,B,Q,R);


T=0:dt:Tf;



xm=x0;
costnew=xm'*Pr*xm+gammaw*dxa*sum(nodesa);
U=[];
Adj=[];
cont=1;
Loc=[];
g=[];

%tic
cl=A-B*Kr;
[~,X]=ode45(@(t,x) cl*x,T,xm);
X=X';
Adj=Pr*X;
U=-R^(-1)*B'*Adj;
g=Adj.*repmat(U,size(Adj,1),1);
%toc

mesh(X)


% tic
% for i=1:length(T)
%     cl=A-cont*B*Kr;
%     xp=(eye(size(A,1))-dt*cl)\xm;
%     adj=Pr*xm;
%     u=(-R^(-1)*B'*adj);
%     U=[U u];
%     adj=-adj; %kevin
%     Adj=[Adj adj];
%     gi=-adj*u;
%     g=[g gi];
%     xm=xp;
% end
% toc
gtot=sum(g(1:N,:),2)*dt+gammaw;

% gtott=0*nodes;
% 
% for ii=1:N
%     gtott=gtott+gtot(ii)*sin(2*pi*nodes*ii);
% 
% end
% 
% gtot=gtott';
    
%ngtot=sqrt(gtot'*M*gtot);
ngtot=sqrt(gtot'*gtot);
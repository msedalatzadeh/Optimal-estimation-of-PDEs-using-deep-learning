function [gtot,ngtot,costnew]=ocsolvenew2d(c,N,nodesa,gamma,gammaw)
[A,B,Q,x,y,h]=heatssfd2d(c,N,nodesa);

R=gamma;
Tf=100;
dt=0.1;
[Kr,Pr,~ ]=lqr(A,B,Q,R);



T=0:dt:Tf;
xinit=0*sin(pi*x).*sin(pi*y)+1;

xm=xinit;
costnew=xm'*Pr*xm+gammaw*h*sum(nodesa);
U=[];
Adj=[];
cont=1;
Loc=[];
g=[];
cli=inv(A-cont*dt*B*Kr);
for i=1:length(T)
    %cl=A-cont*B*Kr;
    %xp=(eye(size(A,1))-dt*cl)\xm;
    xp=cli*xm;
    adj=Pr*xm;
    u=(-R^(-1)*B'*adj);
    U=[U u];
    adj=-adj; %kevin
    Adj=[Adj adj];
    gi=-adj*u;
    g=[g gi];
    xm=xp;
    i*dt
end
gtot=sum(g,2)*dt+gammaw;
%ngtot=sqrt(gtot'*M*gtot);
ngtot=sqrt(gtot'*Q*gtot);
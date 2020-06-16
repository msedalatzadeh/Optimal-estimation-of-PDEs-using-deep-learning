function [gtota,ngtot,costnew]=ocsolve2dmodal(c,n,I,dxa,na,nodesa,gammau,gammaw,costold)
global A M K y0p vol
[B]=ssmodal2dB(I,n,c,nodesa,dxa,M);



R=gammau;
Tf=1000;
dt=.1;
%[Kr,Pr,E]=lqr(A,B,M,R);

[Pr,~,Kr,Rep] = care(A,B,M,R,0*B,eye(n));
if (Rep==-2 || max(eig(A-B*Kr)>=0))
    Rep
    costnew=1e10;
    gtota=zeros(na,1);
    ngtot=1;
else

%T=0:dt:Tf;


% xm=y0p/sqrt(y0p'*y0p);
% xm=M\xm;

   [maxv,~]=eig(Pr,K);
   xm=maxv(:,end);
   xm=xm/norm(xm);
   xm=M\xm;

%%


%costnew=xm'*Pr*xm+gammaw*dxa^2*sum(nodesa(:,4));
costnew=xm'*Pr*xm+gammaw*(dxa^2*sum(nodesa(:,4))-vol)^2;
if (costnew>costold)
        costnew=1e10;
    gtota=zeros(na,1);
    ngtot=1;
else

[T,X]=ode15s(@(t,x) (A-B*Kr)*x,0:dt:Tf,xm);
X=X';
adj=Pr*X;
U=(-R^(-1)*B'*adj);
Adj=-adj;
g=-Adj.*repmat(U,n,1);
gtot=sum(g,2)*dt;




%% defining gtot ver the actuator mesh
gtota=zeros(na,1);

    for j=1:n
        gtota=gtota+gtot(j)*basesin(nodesa(:,2),nodesa(:,3),I(j,2),I(j,3));
    end
    
    %gtota=gtota+gammaw;
    gtota=gtota+2*gammaw*(dxa^2*sum(nodesa(:,4))-vol);
    ngtot=sqrt(sum(gtota.^2))*dxa;
end
end


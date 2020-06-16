function [xnodepi,voldpi] = twodimpi(deltax,Tg)

ncont=64;
xnode=deltax:deltax:1-deltax;
[X,Y]=ndgrid(xnode);

h=1*deltax;
beta=exp(-h);
dim=length(xnode);
Vold=ones(dim,dim);
Vold(Tg==1)=0;


for i=1:ncont
Acontx(i)=cos(2*pi*(i-1)/ncont);
Aconty(i)=sin(2*pi*(i-1)/ncont);
end

Acontx(ncont+1)=0;
Aconty(ncont+1)=0;
ncont=ncont+1;

T=zeros(dim,dim,ncont);
tol=1/5*deltax^2;


aoldx=1*ones(dim,dim);
aoldx([1 dim],:)=0;
aoldx(:,[1 dim])=0;

aoldy=-0*ones(dim,dim);
aoldy([1 dim],:)=0;
aoldy(:,[1 dim])=0;

Varr=zeros(dim,dim,ncont);
diff1=100;


while(diff1>tol)  
    VI=Vold;
    diff=1;

Arrx=X+h*aoldx.*(Tg==0);
Arry=Y+h*aoldy.*(Tg==0);

while (diff>tol)
   arrinterp=interpn(X,Y,Vold,Arrx,Arry,'linear',100);
   Vnew=(beta*arrinterp+1-beta).*(Tg==0);
   diff=max(max(abs(Vnew-Vold))); 
   Vold=Vnew;
end

    for k=1:ncont
        arrx=X+h*Acontx(k).*(Tg==0);
        arry=Y+h*Aconty(k).*(Tg==0);
        arrinterp=interpn(X,Y,Vold,arrx,arry,'linear',100);
        Varr(:,:,k)=arrinterp;
    end
    [~,donde]=min(Varr,[],3);

    for i=1:ncont
        aoldx(donde==i)=Acontx(i);
        aoldy(donde==i)=Aconty(i);
    end


aoldx(Tg==1)=0;
aoldy(Tg==1)=0;




    diff1=max(max(abs(VI-Vold)));

end
xnodepi=xnode;
voldpi=Vold;


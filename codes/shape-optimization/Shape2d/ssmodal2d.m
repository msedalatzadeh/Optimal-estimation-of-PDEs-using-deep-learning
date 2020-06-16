function [A,B,M,K]=ssmodal2d(I,n,c,nodesa,dxa)
M=zeros(n,n);

for i=1:n
    for j=1:n
        Ii=I(i,:);
        Ij=I(j,:);
        if (Ii(2:3)==Ij(2:3))
            M(i,j)=integral2(@(x,y) basesin(x,y,Ii(2),Ii(3)).^2,0,1,0,1);
        end
    end
end

Reac=zeros(n,n);
for i=1:n
    for j=1:n
        Ii=I(i,:);
        Ij=I(j,:);
        %if (Ii(2:3)==Ij(2:3))
            %Reac(i,j)=integral2(@(x,y) reac(x,y).*basesin(x,y,Ii(2),Ii(3)).*basesin(x,y,Ij(2),Ij(3)),0,1,0,1);
            Reac(i,j)=dxa.^2.*sum(reac(nodesa(:,2),nodesa(:,3)).*basesin(nodesa(:,2),nodesa(:,3),Ii(2),Ii(3)).*basesin(nodesa(:,2),nodesa(:,3),Ij(2),Ij(3)));
        %end
    end
end


K=zeros(n,n);

for i=1:n
    for j=1:n
        Ii=I(i,:);
        Ij=I(j,:);
        if (Ii(2:3)==Ij(2:3))
            K(i,j)=integral2(@(x,y) basesinpx(x,y,Ii(2),Ii(3)).^2+basesinpy(x,y,Ii(2),Ii(3)).^2,0,1,0,1);
        end
    end
end


D=zeros(n,n);

for i=1:n
    for j=1:n
        Ii=I(i,:);
        Ij=I(j,:);
        if (Ii(2:3)==Ij(2:3))
            D(i,j)=-c*integral2(@(x,y) difu(x,y).*(basesinpx(x,y,Ii(2),Ii(3)).^2+basesinpy(x,y,Ii(2),Ii(3)).^2),0,1,0,1);
        end
    end
end


A=M\D+M\Reac;

B=zeros(n,1);


for i=1:n
    B(i,1)=dxa^2*sum(basesin(nodesa(:,2),nodesa(:,3),I(i,2),I(i,3)).*nodesa(:,4));
end
B=M\B;
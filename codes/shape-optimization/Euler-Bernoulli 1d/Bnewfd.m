function [B]=Bnewfd(N,nodesa,dxa,Bo)

B1=nodesa;
B=[0*B1;Bo*B1];
function [val]=lapreg(delta,f)
n=length(f);
h=1/(n+1);
D=(delta/h^2)*full(gallery('tridiag',n,-1,2,-1));
val=(D+eye(n))\f;


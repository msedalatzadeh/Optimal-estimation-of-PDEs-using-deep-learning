function [val]=basesinpy(x,y,ix,iy)

val=pi*iy*sin(pi*ix*x).*cos(pi*iy*y);
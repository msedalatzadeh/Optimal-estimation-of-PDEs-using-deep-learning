function [val]=basesinpx(x,y,ix,iy)

val=pi*ix*cos(pi*ix*x).*sin(pi*iy*y);
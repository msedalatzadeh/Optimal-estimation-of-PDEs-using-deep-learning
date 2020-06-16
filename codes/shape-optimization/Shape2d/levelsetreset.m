function [phi0]=levelsetreset(dx,Tg)
tic
% [~, vout] = twodimtestvi(dx,Tg);
% [~, vin] = twodimtestvi(dx,1-Tg);
% phi0=-log(1-vout)+log(1-vin);


[~, vout] = twodimpi(dx,Tg);
[~, vin] = twodimpi(dx,1-Tg);
phi0=-log(1-vout)+log(1-vin);
toc
function [THcode] = TH(Nh,Np)   % 产生TH码
% Np:跳时码周期,周期循环一组c
% Nh:跳时码最大上界   0<c<Nh
THcode = floor(rand(1,Np).*Nh); % 产生Np个c值
end


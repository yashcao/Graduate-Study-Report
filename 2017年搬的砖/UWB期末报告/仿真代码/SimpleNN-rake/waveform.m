function [w]= waveform(fc,Tm,tau)  % 产生功率归一化的脉冲波形
% 高斯波形的二阶导数
dt = 1 / fc;
OVER = floor(Tm/dt);
e = mod(OVER,2);
kbk = floor(OVER/2);
tmp = linspace(dt,Tm/2,kbk);
% w(t+△) = [1-4pi(t/tau)^2] exp[-2pi(t/tau)^2]
s = (1-4.*pi.*((tmp./tau).^2)).* exp(-2.*pi.*((tmp./tau).^2));
y = zeros(1,OVER);
if e                     % 奇数
    for k=1:length(s)
        y(kbk+1)=1;
        y(kbk+1+k)=s(k);
        y(kbk+1-k)=s(k);
    end
else                     % 偶数
    for k=1:length(s)
        y(kbk+k)=s(k);
        y(kbk+1-k)=s(k);
    end
end
E = sum((y.^2).*dt);
w = y ./ (E^0.5);        % 功率归一化
end

%%%%% 参数说明 采集 35 个数据点 %%%%%%
% 'fc' ：抽样频率     50e9
% 'Tm' ：脉冲持续时间 0.7e-9
% 'tau' ：成形参数    0.2877e-9


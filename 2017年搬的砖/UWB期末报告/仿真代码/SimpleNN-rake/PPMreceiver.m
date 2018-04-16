% 构造 2PPM TH UWB 信号的接收机，并计算平均错误率BER
function [RXbits,BER] = PPMreceiver(R,mask,fc,bits,numbit,Ns,Ts)
HDSD = 1;
% HDSD = 1 --> 硬判决，接收机对表示一个比特的Ns个脉冲逐一独立判断。
%              将超过门限的脉冲数与低于门限的脉冲数比较，输出较多者对应的比特
% HDSD = 2 --> 软判决，接收机把Ns个脉冲形成的信号当作一个单独的多脉冲信号
%              接收机与信号进行相关的掩膜m(t)是整个比特的脉冲串
% 软判决的效果要好于硬判决

% 实现相关器
% 一个信号波形一行，共有N个波形。每个波形长度为L，即每个波形的样本数
[N,L] = size(R);
RXbits = zeros(N,numbit);
dt = 1 / fc;
framesamples = floor(Ts ./ dt);
bitsamples = framesamples * Ns;

for n = 1 : N                 % 分别取出N个波形，逐一判断
    rx = R(n,:);
    mx = rx .* mask;

    if HDSD == 1              % 硬判决
        for nb = 1 : numbit
            mxk = mx(1+(nb-1)*bitsamples : bitsamples+(nb-1)*bitsamples);
            No0 = 0;
            No1 = 0;
            for np = 1 : Ns
                mxkp = mxk(1+(np-1)*framesamples : framesamples+(np-1)*framesamples);
                zp = sum(mxkp.*dt);
                if zp > 0     % 积分周期为Ts
                    No0 = No0 + 1;
                else
                    No1 = No1 + 1;
                end
            end
            if No0 > No1
                RXbits(n,nb) = 0;
            else
                RXbits(n,nb) = 1;
            end
        end % for nb = 1 : numbit
    end % end of Hard Decision Detection
%{    
    if HDSD == 2 % 软判决
        for nb = 1 : numbit
            mxk = mx(1+(nb-1)*bitsamples:bitsamples+ (nb-1)*bitsamples);
            zb = sum(mxk.*dt);
            if zb > 0  % 积分周期为NsTs
                RXbits(n,nb) = 0;% Z>0，判断为0
            else
                RXbits(n,nb) = 1;% Z<0，判断为1
            end
        end % for nb = 1 : numbit
    end % end of Soft Decision Detection
 %}   
end % for n = 1 : N
for n = 1 : N % 计算误比特率
    WB = sum(abs(bits-RXbits(n,:)));
    BER(n) = WB / numbit;
end
end

% 'R'： 表示所使用的波形矩阵，一个波形对应于矩阵的一行
% 'mask'：表示相关掩膜 
% 'fc'：抽样频率
% 'bits' 发射机产生的原始二进制比特流
% 'Ns' 每比特的脉冲数（即用几个脉冲表示1比特）
% 'Ts' 平均脉冲重复周期，即一帧的长度
% 函数返回：
% 'RXbits' ：存储经解调后的二进制数据流
% 'BER'：存储计算得到的Prb直（误比特率）

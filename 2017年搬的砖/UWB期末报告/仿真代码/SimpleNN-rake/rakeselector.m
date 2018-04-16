function [G,T,NF,Arake,Srake,Prake] =rakeselector(hf,fc,ts,L,S)
%% 构造rake接收机的分集部分，对离散时间信道模型进行信道估值
dt = 1 / fc;
ahf = abs(hf);
[s_val,s_ind] = sort(ahf);
NF = 0;                  % 非零分量个数初始化为0
i = length(s_ind);       % 系数的个数，即非零分量个数的最大值
j = 0;
while (s_val(i)>0)&(i>0)
    NF = NF + 1;
    j = j + 1;
    index = s_ind(i);    % 排序结束后，从尾部即最大处往前操作
    I(j) = index;
    T(j) = (index-1)*dt;
    G(j) = hf(index);
    i = i - 1;
end
%% 计算权重
binsamples = floor(ts/dt);
if S > NF                % 'S' ：Srake用到的分量数
    S = NF;              % 'NF'：信道冲激响应中非零指的个数
end

if L > NF
    L = NF;
end
Arake = zeros(1,NF*binsamples);
Srake = zeros(1,NF*binsamples);
Prake = zeros(1,NF*binsamples);
% Selective Rake and All Rake
for nf = 1 : NF
    x = I(nf);y = G(nf);
    Arake(x) = y;
    if nf <= S
        Srake(x) = y; % 只输出S个最大的多径分量
    end
end % for nf = 1 : NF
% PRake
[tv,ti] = sort(T);
TV = tv(1:L);
TI = ti(1:L);
tc = 0;
for nl = 1 : length(TV)
    index = TI(nl);
    x = I(index);
    y = G(index);
    Prake(x) = y;
    tc = tc + 1;
    L = L - 1;
end
end

% 输入参数：
% hf：信道冲激相应
% ts：时间仓长度
% 'fc'：抽样频率
% 'S' ：Srake用到的分量数
%　函数返回：
% 1) 'G'：包含信道冲激响应的所有幅度系数，降序排列。
% 2)  'T'：针对G中的各个分量相应的到达时间。
% 3)  'NF'：信道冲激响应中非零指的个数
% 4) 'Srake'：包含Selective RAKE中用到的权重因子

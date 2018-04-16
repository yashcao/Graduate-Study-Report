% 根据IEEE 802.15.SG3a.产生信道冲激响应 
% TMG:信道总多径增益
function [h0,hf,OT,ts,X] = UWBC(fc,TMG)  % UWBC1
OT = 200e-9;                             % 观测时间 [s]
ts = 1e-9;                               % 离散分辨率 [s]
LAMBDA = 0.0223*1e9;                     % 簇平均到达因子 (1/s)
lambda = 2.5e9;                          % 簇内脉冲平均到达因子 (1/s)
GAMMA = 7.1e-9;                          % 簇衰减因子
gamma = 4.3e-9;                          % 簇内脉冲衰减因子
sigma1 = 10^(3.3941/10);                 % 簇的信道衰减系数偏差
sigma2 = 10^(3.3941/10);                 % 簇内脉冲信道衰减系数偏差
sigmax = 10^(3/10);                      % 信道幅度增益的标准偏差
% 脉冲衰减阈值，当exp(-t/gamma)<rdt时，该脉冲忽略
rdt = 0.001;
% 峰值阈值 [dB]，只考虑幅度在峰值-PT范围以内的脉冲
PT = 50;

% 簇的形成
dt = 1 / fc;    % 采样频率
T = 1 / LAMBDA; % 簇平均到达时间
t = 1 / lambda; % 簇内脉冲平均到达时间[s]
i = 1;
CAT(i)=0;       % 第一簇到达时间，初始化为0
next = 0;
while next < OT
    i = i + 1;
    next = next + expinv(rand,T); % 产生簇的到达时间，服从p(Tn/Tn-1)=lambda*[-exp(Tn/Tn-1)]
    if next < OT
        CAT(i)= next;
    end
end
% 路径
NC = length(CAT); % 参考的簇数
logvar = (1/20)*((sigma1^2)+(sigma2^2))*log(10);
omega = 1;
pc = 0;           % 多径数量计数器
for i = 1 : NC
    pc = pc + 1;
    CT = CAT(i);
    HT(pc) = CT;
    next = 0;
    mx = 10*log(omega)-(10*CT/GAMMA);
    mu = (mx/log(10))-logvar;
    a = 10^((mu+(sigma1*randn)+(sigma2*randn))/20);
    HA(pc) = ((rand>0.5)*2-1).*a;
    ccoeff = sigma1*randn; % 簇衰减
    while exp(-next/gamma)>rdt
        pc = pc + 1;
        next = next + expinv(rand,t);
        HT(pc) = CT + next;
        mx = 10*log(omega)-(10*CT/GAMMA)-(10*next/GAMMA);
        mu = (mx/log(10))-logvar;
        a = 10^((mu+ccoeff+(sigma2*randn))/20);
        HA(pc) = ((rand>0.5)*2-1).*a;
    end
end
peak = abs(max(HA)); % 峰值滤波器
limit = peak/10^(PT/10);
HA = HA .* (abs(HA)>(limit.*ones(1,length(HA))));
%凡小于limit的脉冲不输出
for i = 1 : pc
    itk = floor(HT(i)/dt);
    h(itk+1) = HA(i);
end
% 离散相应形式
N = floor(ts/dt);
L = N*ceil(length(h)/N);
h0 = zeros(1,L);
hf = h0;
h0(1:length(h)) = h;
for i = 1 : (length(h0)/N)
    tmp = 0;
    for j = 1 : N
        tmp = tmp + h0(j+(i-1)*N);
    end
    hf(1+(i-1)*N) = tmp;
end
E_tot=sum(h.^2); % 功率归一化
h0 = h0 / sqrt(E_tot);
E_tot=sum(hf.^2);
hf = hf / sqrt(E_tot);
mux = ((10*log(TMG))/log(10)) - (((sigmax^2)*log(10))/20);
X = 10^((mux+(sigmax*randn))/20);
h0 = X.*h0;
hf = X.*hf;
%% 图形输出
G = 0;
if G 
    Tmax = dt*length(h0);
    time = (0:dt:Tmax-dt);
    figure(1);
    S1=stem(time,h0);
    AX=gca;
    set(AX,'FontSize',12);
    T=title('连续时间信道冲激响应');
    set(T,'FontSize',12);
    x=xlabel('时间 [s]');
    set(x,'FontSize',12);
    y=ylabel('幅度增益');
    set(y,'FontSize',12);
    figure(2);
    S2=stairs(time,hf);
    AX=gca;
    set(AX,'FontSize',12);
    T=title('离散时间信道冲激响应');
    set(T,'FontSize',12);
    x=xlabel('时间 [s]');
    set(x,'FontSize',12);y=ylabel('幅度增益');
    set(y,'FontSize',12);
end
end
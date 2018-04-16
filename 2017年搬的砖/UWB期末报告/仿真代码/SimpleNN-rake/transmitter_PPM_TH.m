function [bits,THcode,Stx,ref]=transmitter_PPM_TH
Pow = -30;
fc = 50e9;
numbits = 3;
Ns = 1;
Tc = 1e-9; %一个时隙长度
Nh = 5;
Ts = 5e-9;
Np = 2000;
Tm = 0.5e-9;
tau = 0.25e-9;
dPPM = 0.5e-9;
bits = bit(numbits); % 产生numbits个比特的原始信号（0/1等概）
repbits = repcode(bits,Ns); % 每个比特位重复编码为Ns位(帧)
THcode = TH(Nh,Np); % 每帧划分Nh个chip，每Np个c循环一次
[PPMTHseq,THseq] = PPM_TH(repbits,fc,Tc,Ts,dPPM,THcode);
%% 成形滤波
power = (10^(Pow/10))/1000; % Pow:传输功率
Ex = power * Ts;
w0 = waveform(fc,Tm,tau); 
wtx = w0 .* sqrt(Ex);
Sa = conv(PPMTHseq,wtx);
Sb = conv(THseq,wtx);
%% 产生输出信号
L = (floor(Ts*fc))*Ns*numbits;
% L = floor((Ts*fc)*Ns*numbits);
Stx = Sa(1:L);
ref = Sb(1:L);
%% 绘图开关
G = 0; 
opengl hardware;
if G
    F = figure(4);
    set(F,'Position',[32 223 951 420]); % 设置绘图的大小比例
    tmax = numbits*Ns*Ts;
    time = linspace(0,tmax,length(Stx));
    P = plot(time,Stx);
    set(P,'LineWidth',[2]);
    ylow=-1.5*abs(min(wtx));yhigh=1.5*max(wtx);
    axis([0 tmax ylow yhigh]);
    AX=gca;
    set(AX,'FontSize',12);
    X=xlabel('Time [s]');
    set(X,'FontSize',14);
    Y=ylabel('Amplitude [V]');
    set(Y,'FontSize',14);
    for j = 1 : numbits
        tj = (j-1)*Ns*Ts;
        L1=line([tj tj],[ylow yhigh]);
        set(L1,'Color',[0 0 0],'LineStyle','--','LineWidth',[2]); % 间隔原始比特的线
        for k = 0 : Ns-1
            if k > 0
            tn = tj + k*Nh*Tc;
            L2=line([tn tn],[ylow yhigh]);
            set(L2,'Color',[0.5 0.5 0.5],'LineStyle','-.','LineWidth',[2]); % 间隔符号位的线
            end
            for q = 1 : Nh-1
                th = tj + k*Nh*Tc + q*Tc;
                L3=line([th th],[0.8*ylow 0.8*yhigh]);
                set(L3,'Color',[0 0 0],'LineStyle',':','LineWidth',[1]); % 间隔时隙的线
            end
        end
    end
end
end
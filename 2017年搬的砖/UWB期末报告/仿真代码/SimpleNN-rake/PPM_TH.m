function [PPMTHseq,THseq] = PPM_TH(seq,fc,Tc,Ts,dPPM,THcode)
% 引入TH码并进行PPM调制
dt = 1 ./ fc;
framesamples = floor(Ts./dt);   % 每个脉冲周期的样本数
chipsamples = floor (Tc./dt);   % 每个帧周期的样本数
PPMsamples = floor (dPPM./dt);  % PPM时移内的采样数
THp = length(THcode);           % 跳时码循环周期
totlength = framesamples*length(seq);
PPMTHseq=zeros(1,totlength);
THseq=zeros(1,totlength);
for k = 1 : length(seq)         % 引入TH码和PPM %s(t)=sum(p(t-jTs-CjTc-aE))
    % 脉冲位置，表示第k个脉冲-jTs，共len(seq)个脉冲
    index = 1 + (k-1)*framesamples;
    % 引入TH码,-CjTc，表示第几个时隙
    kTH =THcode(1+mod(k-1,THp));
    index = index + kTH*chipsamples;
    THseq(index) = 1;
    % 引入PPM时移,-aE，表示在时隙内的位置
    index = index + PPMsamples*seq(k);
    PPMTHseq(index) = 1;
end
end
%%%%% 参数说明 %%%%%
% 'seq'：二进制源码 
% 'fc' ：抽样频率 = 50e9
% 'Tc' ：时隙，一个chip的长度 = 1e-9
% 'Ts' ：脉冲平均重复周期 = Nh * Tc = 3e-9
% 'dPPM'：脉位调制d，PPM引入的时移 = 0.5e-9
% 'THcode' ：TH码
% 产生两个输出：
% '2PPMTHseq' ：TH和PPM共同调制信号
% 'THseq' ：未经PPM调制的信号




% 为二进制PPM UWB信号的Rake接收机计算相关掩膜mask
function [mask] = PPMcorrmask_R(ref,fc,numpulses,dPPM,rake)
dt = 1 / fc;
LR = length(ref);
Epulse = (sum((ref.^2).*dt))/numpulses; % 功率归一化
nref = ref./sqrt(Epulse);
mref = conv(nref,rake); % Rake 卷积
mref = mref(1:LR);
PPMsamples = floor (dPPM ./ dt); % 构造相关掩膜
sref(1:PPMsamples)=mref(LR-PPMsamples+1:LR);
sref(PPMsamples+1 : LR)=mref(1 : LR-PPMsamples);
mask = mref-sref;
end

% 'ref'：未经PPM调制的参考信号
% 'fc'：抽样频率  
% 'numpulses' ：传输脉冲数目
% 'dPPM'：PPM时移量
% 'rake'：离散冲激相应

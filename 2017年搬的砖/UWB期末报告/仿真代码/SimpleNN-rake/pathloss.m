% 根据给定的距离d，衰减因子gamma以及1米处的信号c0将输入信号衰减
% 函数返回衰减后的信号rx以及信道增益attn
function [rx,attn] = pathloss(tx,c0,d,gamma)
attn = (c0/sqrt(d^gamma));
rx = attn .* tx;
end
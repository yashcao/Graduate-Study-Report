function [bits] = bit(numbits)  % 产生二进制原信号
% 原信号比特数numbis作为输入
% rand产生的是在0～1上均匀分布的随机数
% 这些数>0.5的几率各是一半，即bis为0，1的几率各半 
bits=rand(1,numbits)>0.5;
end


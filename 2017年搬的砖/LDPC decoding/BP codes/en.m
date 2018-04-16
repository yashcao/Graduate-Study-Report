clc;
clear all;
%%校验矩阵/生成矩阵
G = [ 1 1 1 1 1 0 0 0 1 0 0 0;
      0 0 1 1 0 0 0 1 0 1 0 0;
      1 1 1 0 1 0 0 1 0 0 1 0;
      1 0 0 1 1 1 0 1 0 0 0 1;];
  
H = [0 1 0 1 0 1 1 1 0 0 0 1;
     1 0 1 1 0 0 0 0 1 0 0 0;
     0 1 0 0 1 0 1 0 0 0 0 1;
     1 0 0 1 0 0 0 0 0 1 1 0;
     0 0 1 0 1 1 0 0 0 1 0 0;
     1 0 1 0 0 0 1 1 0 0 1 0
     0 1 0 0 0 1 0 1 1 1 0 0;
     0 0 0 0 1 0 0 0 1 0 1 1;
     ]; 
%%send orignal info bit
u = [1 1 0 1];
%%信源编码
c0 = u*G;
%%仿制2psk
c1 = 2*rem(c0,2)-1;
disp(c1);
len = length(c1);


%%模拟AWGN
%%noise = zeros(1,len)
%%mu=0 sigma=0.8
noise = normrnd(0,0.8,1,len);
%%接收信号
%%r=zeros(1,len)
r=c1+noise;
spa_decode(c1,r,H);




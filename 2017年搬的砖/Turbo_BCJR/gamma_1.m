% GAMMA(m',m) =
% P(X=i) * P(S(k+1)=m|S(k)=m',X=i) * P(R0(k) R1(k)|S(k+1)=m, S(k)=m', X=i))
% there are 2 columns, 1st for i/p0 and 2nd for i/p1.
% 状态转移格子图
% for 1st column：（0 转移） 00/10/01/11 ===0==>00/01/10/11
% for 2nd column：（1 转移） 00/10/01/11 ===1==>10/11/00/01

% P(X=i)，P(S(k+1)=m|S(k)=m',X=i)，P(R0(k),R1(k)|S(k+1)=m,S(k)=m',X=i))
function [g]=gamma_1(Apriori,N,Input_matrix,Parity_bit_matrix,R0,R1,SNR) 
    
    g=zeros(4,2*N);
    for i=1:N
        T1=((R0(i)-sqrt(SNR)*Input_matrix(1,1))^2)+((R1(i)-sqrt(SNR)*Parity_bit_matrix(1,1))^2);
        T2=((R0(i)-sqrt(SNR)*Input_matrix(1,2))^2)+((R1(i)-sqrt(SNR)*Parity_bit_matrix(1,2))^2);
        T3=((R0(i)-sqrt(SNR)*Input_matrix(2,1))^2)+((R1(i)-sqrt(SNR)*Parity_bit_matrix(2,1))^2);
        T4=((R0(i)-sqrt(SNR)*Input_matrix(2,2))^2)+((R1(i)-sqrt(SNR)*Parity_bit_matrix(2,2))^2);
        T5=((R0(i)-sqrt(SNR)*Input_matrix(3,1))^2)+((R1(i)-sqrt(SNR)*Parity_bit_matrix(3,1))^2);
        T6=((R0(i)-sqrt(SNR)*Input_matrix(3,2))^2)+((R1(i)-sqrt(SNR)*Parity_bit_matrix(3,2))^2);
        T7=((R0(i)-sqrt(SNR)*Input_matrix(4,1))^2)+((R1(i)-sqrt(SNR)*Parity_bit_matrix(4,1))^2);
        T8=((R0(i)-sqrt(SNR)*Input_matrix(4,2))^2)+((R1(i)-sqrt(SNR)*Parity_bit_matrix(4,2))^2);
        
        g(1,2*i-1)= Apriori(1,i)*(1/(2*pi))*exp(-0.5*T1);         %Current State=0(00), i/p0
        g(1,2*i)  = Apriori(2,i)*(1/(2*pi))*exp(-0.5*T2);         %Current State=0(00), i/p1
        g(2,2*i-1)= Apriori(1,i)*(1/(2*pi))*exp(-0.5*T3);         %Current State=2(10), i/p0
        g(2,2*i)  = Apriori(2,i)*(1/(2*pi))*exp(-0.5*T4);         %Current State=2(10), i/p1
        g(3,2*i-1)= Apriori(1,i)*(1/(2*pi))*exp(-0.5*T5);         %Current State=1(01), i/p0
        g(3,2*i)  = Apriori(2,i)*(1/(2*pi))*exp(-0.5*T6);         %Current State=1(01), i/p1
        g(4,2*i-1)= Apriori(1,i)*(1/(2*pi))*exp(-0.5*T7);         %Current State=3(11), i/p0
        g(4,2*i)  = Apriori(2,i)*(1/(2*pi))*exp(-0.5*T8);         %Current State=3(11), i/p1
    end    
end
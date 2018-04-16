%% Turbo Code 
% Encoder: RSC (Recursive Systematic Convolution)
% Decoder: BCJR iterative decoder
% AWGN channel for BPSK modulated symbols
close all;clear all;clc;

%% 参数
N=10; %输入序列长
X=floor(2*rand(1,N)); %发送一段信息位
Interleaver=randperm(N); %交织规则，1-8打乱顺序
X_pi(1:N)=X(Interleaver(1:N)); %生成交织码
SNRdB=0:0.5:9;                  %SNR in dB
SNR=10.^(SNRdB/10);       	    %SNR in linear scale

%% Encoding RSC=[(1+d+d2)/(1+d2)] =>output=[1+d'];d'=1+d2
C0=zeros(1,N); % RSC-0 状态d'寄存[2-N]
C1=zeros(1,N); % RSC-1 状态d'寄存[2-N]
% 根据输入得到d'[2-N]
for i=1:N
    k = i;
    while (k >= 1)
        C0(i) = xor ( C0(i),X(k) );
        C1(i) = xor ( C1(i),X_pi(k) );
        k=k-2;
    end
end
% 补全d';[0,C0(1:end-1)];得到校验位
P0 = xor (X,[0,C0(1:end-1)]);
P1 = xor (X_pi,[0,C1(1:end-1)]);

%% BPSK modulation
mod_code_bit0=2*X-1;        %Modulating Code Bits using BPSK Modulation
mod_code_bit1=2*P0-1;
mod_code_bit2=2*P1-1;

%% AWGN and received codewords[R0 R1 R2]
SNR_k=10; % 暂取一个SNR定值
% randn（mu =0,sigma=1）
R0=sqrt(SNR(SNR_k))*mod_code_bit0+randn(1,N);   % Received Codebits Corresponding to input bits
R1=sqrt(SNR(SNR_k))*mod_code_bit1+randn(1,N);   % Received Codebits Corresponding to parity bits of RSC-0
R2=sqrt(SNR(SNR_k))*mod_code_bit2+randn(1,N);   % Received Codebits Corresponding to parity bits of RSC-1
R0_pi(1:N)=R0(Interleaver(1:N));            % Received Codebits 输入交织

%% Decoding parameter
BCJR=0; % First iteration by BCJR-0
Iteration=5;
% 1-column input=0 and 2-column input=1
Input_matrix=2*[0,1;0,1;0,1;0,1]-1;           
% Each row represents state 00,10,01,11 
Parity_bit_matrix=2*[0,1;1,0;0,1;1,0]-1;       
% 1-row for prob. of i/p0 & 2-row for prob. of i/p1    
Apriori=ones(2,N);          
Apriori=Apriori*0.5; % Initializing all apriori to 1/2

%% observation
disp(X);

%% 迭代译码 Iterative process starts here 
for iter=1:Iteration
    if BCJR==0  %If BCJR is 0 then pass R0 and R1 to calculate GAMMA
            GAMMA=gamma_1(Apriori,N,Input_matrix,Parity_bit_matrix,R0,R1,SNR(SNR_k));
    else        %If BCJR is 1 then pass R0_pi and R2 to calculate GAMMA
            GAMMA=gamma_1(Apriori,N,Input_matrix,Parity_bit_matrix,R0_pi,R2,SNR(SNR_k));
    end
    % 计算 ALPHA,BETA
    ALPHA=alpha_1(GAMMA,N); % Calc ALPHA at each stage using GAMMA and previous ALPHA
    BETA=beta_1(GAMMA,N);   % Calc BETA at each stage using GAMMA and next BETA 
    
    % Calc LAPPR using ALPHA,BETA,GAMMA
    [~,~,LAPPR_1]=lappr(ALPHA,BETA,GAMMA,N);
    decoded_bits=zeros(1,N);
    if BCJR==0
        LAPPR_TEMP=LAPPR_1;
    else
        LAPPR_TEMP(Interleaver(1:N))=LAPPR_1(1:N);
    end
    decoded_bits(LAPPR_TEMP>0)=1;  % Decoding is done using LAPPR values
    
    %% observation
    disp(decoded_bits);
    
    if BCJR==0      % If the decoder is BCJR-0 then
        lappr_2(1:N)=LAPPR_1(Interleaver(1:N));     % Interleave the LAPPR values and pass to BCJR-1
    else            % If the decoder is BCJR-1 then
        lappr_2(Interleaver(1:N))=LAPPR_1(1:N);     % Re-interleave the LAPPR values and pass to BCJR-0
    end
    LAPPR_1=lappr_2;

    Apriori(1,1:N)=1./(1+exp(LAPPR_1));              % Apriori corresponding to input 0
    Apriori(2,1:N)=exp(LAPPR_1)./(1+exp(LAPPR_1));   % Apriori corresponding to input 1
        
    BCJR=~BCJR;   % Changing the state of the decoder for the next iteration
end

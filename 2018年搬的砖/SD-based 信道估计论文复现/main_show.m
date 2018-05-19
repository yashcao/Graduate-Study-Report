clear all;
clc;
SNR_dB=[0:5:40];  
SNR_linear=10.^(SNR_dB/10.);
N_iter=200; 
n = 256; % number of beams (transmit antennas)
K = 16; % number of users
Q = K*6;
L = 3; % number of paths per user
V = 8; % Retained number of elements for each component
lamada = 1; % wavelength
N = K; % number of retained RF chains
d = lamada/2;
NMSE = zeros(1,length(SNR_dB));
% sigma2=1/SNR_linear(5);
for i_snr=1:length(SNR_dB) 
    i_snr
    SNR=SNR_linear(i_snr);
    sigma2=1/SNR_linear(i_snr);
    temp = 0; temp1 = 0; temp2 = 0; temp3 = 0; temp4 = 0; temp5 = 0; temp6 = 0;
    error1 = 0; error2 = 0; error3 = 0; error4 = 0; 
    for iter = 1:N_iter
        H = beamspace_channel(n,K,L); % generate the signalspace channel
        U = zeros(n,n); 
        deta = 1/n;
        for i = -(n-1)/2:1:(n-1)/2
            U(:,i+(n+1)/2) = sqrt(1/n)*exp(1i*[0:n-1]*2*pi*deta*i).';
        end
        H_beam = U.'*H; % beamspace channel
        for j = 1 : K
            h = H_beam(:,j);
            % Phi = exp(1i*2*pi*rand(6*K,n));  % analog beamforming Q=6*16 µ¼ÆµÐòÁÐ
            Phi = rand(6*K,n); Phi = Phi>0.5; Phi=2*Phi-1; % Phi = Phi/(sqrt(Q));
            
            noise = sqrt(sigma2)*(randn(6*K,1)+1i*randn(6*K,1))/sqrt(2);
            x = Phi*h + noise;
            [h_hat1, support1, used_iter1] = OMP(x, Phi, 24);
            error1 = error1 + norm(h - h_hat1, 2)^2/norm(h,2)^2;
            [h_hat2, support2] = SD_based(x, Phi, L, V);
            error2 = error2 + norm(h - h_hat2, 2)^2/norm(h,2)^2;
%             [h_hat3, support3, used_iter3] = CoSaMP(x,Phi,24);
%             error3 = error3 + norm(h - h_hat3, 2)^2/norm(h,2)^2;
        end
        %%% ZF precoding
        %%% Full digital
        F = H_beam*inv(H_beam'*H_beam);
        beta = sqrt(K/trace(F*F'));
        H_eq=H_beam'*F;
        for k=1:K
            sum_inf=sum(abs(H_eq(:,k)).^2)-abs(H_eq(k,k))^2;
            temp=temp+log2(1+abs(H_eq(k,k))^2/(sum_inf+K/(SNR*beta^2)));
        end
    end
    NMSE1(i_snr) = error1/K/N_iter;
    NMSE2(i_snr) = error2/K/N_iter;  
    % NMSE3(i_snr) = error3/K/N_iter;
end
 
figure(1)
% semilogy(SNR_dB,NMSE1,'b','Linewidth',1.5);
% hold on 
% semilogy(SNR_dB,NMSE2,'r','Linewidth',1.5);

semilogy(SNR_dB,NMSE1,'b','Linewidth',1,'Marker','s');
hold on 
semilogy(SNR_dB,NMSE2,'r','Linewidth',1,'Marker','d');
% hold on

% semilogy(SNR_dB,NMSE3,'g','Linewidth',1,'Marker','*');
grid on
xlabel('SNR (dB)');
ylabel('NMSE (dB)');
% contourf(abs(H_beam'));
leg = legend('OMP-based','SD-based');
% leg = legend('OMP-based','SD-based','CoSaMP');
set(leg,'Location','NorthEast') 
% contourf(abs(H_beam'));
%{
figure(2)
plot(SNR_dB,sum_rate,'m','LineStyle','--','Linewidth',1,'Marker','o');
grid on
xlabel('SNR (dB)');
ylabel('sum rate (bits/s/Hz)');
%}
function [H]=beamspace_channel(n,K,L)
% n: number of transmit beams (transmit antennas)
% K: number of users
% L: number of paths, L=1 for LoS, L>1 for multipath
lamada = 1; % wavelength
d = lamada/2; % distance between two antennas
H = zeros(n,K);
theta=zeros(L,K);
for k=1:K
    beta=zeros(1,L); 
    %%% complex gain
%     beta(1:L) = (randn(1,L)+1i*randn(1,L))/sqrt(2);
%     beta(1) = 1; % gain of the LoS
%     beta(2:L) = 10^(-0.5)*(randn(1,L-1)+1i*randn(1,L-1))/sqrt(2);
    
    beta(1) = (randn(1)+1i*randn(1))/sqrt(2);
    beta(2:L) = sqrt(10^(-0.5))*(randn(1,L-1)+1i*randn(1,L-1))/sqrt(2);
%     beta(2:L) = sqrt(0.1*beta(1));
%     beta(2:L) = sqrt(0.1*beta(1))*exp(-1i*2*pi*rand(1,L-1)); % gain of
%     NLoS
    %%% DoA
    theta(1,k) = pi*rand(1) - pi/2;
    theta(2:L,k) = pi*rand(1,L-1) - pi/2;
%     while (abs(theta(2)-theta(1))< pi/10) || (abs(theta(2)-theta(3))< pi/10) || (abs(theta(3)-theta(1))< pi/10)
%         theta(2:L,k) = pi*rand(1,L-1) - pi/2;
%     end
    %%% channel for the k th user
    for j = 1:L
        H(:,k) = H(:,k) + beta(j)*array_respones(theta(j,k),n,d,lamada);
    end
end
% 0.5*sin(theta);
% H = sqrt(n/L)*H;
% H = sqrt(1/L)*H;
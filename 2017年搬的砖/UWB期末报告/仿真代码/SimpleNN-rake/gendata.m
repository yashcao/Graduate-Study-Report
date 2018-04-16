% clear; close all; clc; opengl hardware;
function [bitsdata,RXdata] = gendata
% ABER_sum = 0;
% CBER_sum = 0;
% EBER_sum = 0;
bitsdata = [];
RXdata = [];
epochs = 1000;
% epochs=1;
for iteration=1:epochs
[bits,THcode,Stx,ref]= transmitter_PPM_TH;
tx=1; c0=10^(-47/20); % 路径损耗47dB 
d=2; % 接收机与发射机相距2m
gamma=1.7; % 衰减指数
[rx,ag]= pathloss(tx,c0,d,gamma);
TMG=ag^2;
fc=50e9;
[h0,hf,OT,ts,X]= UWBC(fc,TMG);
%[PDP] = PDP(h0,fc);
SRX0=Stx.*ag;
numpulses = 15; % numbits * Ns
%exno=[0 2 4 6];
%exno=[0 2 4 6 8 10];
exno = 4;
[output,noise]= Gnoise(SRX0,exno,numpulses);
SRX=conv(Stx,hf);
SRX=SRX(1:length(Stx));
RX(1,:)=SRX+noise(1,:);
%{
RX(2,:)=SRX+noise(2,:);
RX(3,:)=SRX+noise(3,:);
RX(4,:)=SRX+noise(4,:);

RX(5,:)=SRX+noise(5,:);
RX(6,:)=SRX+noise(6,:);
%}
%{
L=10;
S=10;
[G,T,NF,rec_A,rec_B,rec_D]= rakeselector(hf,fc,ts,L,S);

L=2;
S=2;
[G,T,NF,rec_A,rec_C,rec_E]= rakeselector(hf,fc,ts,L,S);

dPPM=0.5e-9;
[mask_A]= PPMcorrmask_R(ref,fc,numpulses,dPPM,rec_A);
[mask_B]= PPMcorrmask_R(ref,fc,numpulses,dPPM,rec_B);
[mask_C]= PPMcorrmask_R(ref,fc,numpulses,dPPM,rec_C);
[mask_D]= PPMcorrmask_R(ref,fc,numpulses,dPPM,rec_D);
[mask_E]= PPMcorrmask_R(ref,fc,numpulses,dPPM,rec_E);

numbit = 10;
Ns = 3;
Ts = 5e-9;

[RXbits,ABER]= PPMreceiver(RX,mask_A,fc,bits,numbit,Ns,Ts);
[RXbits,BBER]= PPMreceiver(RX,mask_B,fc,bits,numbit,Ns,Ts);
[RXbits,CBER]= PPMreceiver(RX,mask_C,fc,bits,numbit,Ns,Ts);
[RXbits,DBER]= PPMreceiver(RX,mask_D,fc,bits,numbit,Ns,Ts);
[RXbits,EBER]= PPMreceiver(RX,mask_E,fc,bits,numbit,Ns,Ts);
%}
% ABER_sum = ABER_sum + ABER;
% CBER_sum = CBER_sum + CBER;
% EBER_sum = EBER_sum + EBER;
bitsdata = [bitsdata;bits];
% RXdata = cat(1,RXdata,RX);
RXdata = cat(1,RXdata,Stx);
end




%{
ABER = ABER_sum/epochs;
CBER = CBER_sum/epochs;
EBER = EBER_sum/epochs;

figure(3);
semilogy(exno,ABER,'-x',exno,CBER,'-o',exno,EBER,'-s','linewidth',1.5);
legend('理想Rake','SRake','PRake');
xlabel('Ex/N0 (dB)'); 
ylabel('BER'); 
title('BER Performance On PPM UWB System'); 
axis([0 10 10^(-4) 1]);
grid on;
%}
end

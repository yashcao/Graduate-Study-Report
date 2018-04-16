function [repbits] = repcode(bits,Ns)  % 产生重复编码  
% Ns ：码元重复数
numbits = length(bits);
temprect=ones(1,Ns);
temp1=zeros(1,numbits*Ns);
temp1(1:Ns:1+Ns*(numbits-1))=bits;
temp2=conv(temp1,temprect);
repbits=temp2(1:Ns*numbits);
end


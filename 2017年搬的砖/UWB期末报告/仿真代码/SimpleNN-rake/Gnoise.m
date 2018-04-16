function [output,noise] =Gnoise(input,exno,numpulses) 
Ex = (1/numpulses)*sum(input.^2); % 一个单脉冲的平均接收能量
ExNo = 10.^(exno./10);
No = Ex ./ ExNo;
nstdv = sqrt(No./2);              % 噪声的标准差
for j = 1 : length(ExNo)
    noise(j,:) = nstdv(j) .* randn(1,length(input));
    output(j,:) = noise(j,:) + input;
end
end
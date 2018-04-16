%计算R0(2)/R1(2)。穷举2^5=32种1/0组合
%j=0:4按位与 实际上就是取每种组合的1的个数
function [R0,R1] = CalcR(h,q0,q1)
R0 = 0;
R1 = 0;
for i=0:2^(length(h))-1
    tmp = 1;
    k = 0;
    %第j+1位是1是0
    for j = 0:length(h)-1
        if(bitand(2^j,i))
            tmp = tmp*q1(h(j+1));
            k = k+1;
        else
            tmp = tmp*q0(h(j+1));
        end
    end
    %k：1的个数
    if(mod(k,2)==1)
        R1 =  R1 + tmp;
    else
        R0 =  R0 + tmp;
    end
end
end
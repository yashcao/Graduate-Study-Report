function [Q0,Q1] = CalcQ2(h,f0,f1,r0,r1)
Q0 = zeros(1,length(r0));
Q1 = zeros(1,length(r0));
for i = 1:length(h)
    %v send c[i] info by all c_nodes excpet c[i]
    [Q0(h(i)),Q1(h(i))] = CalcQ([h(1:i-1),h(i+1:end)],f0,f1,r0,r1);
end
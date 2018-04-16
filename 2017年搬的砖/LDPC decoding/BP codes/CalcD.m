%最终的决策概率
function [d] = CalcD(h,f0,f1,r0,r1)
d0 = f0;
d1 = f1;
%涉及到该v_node的所有c_node判断的概率
for i = 1:length(h)
    d0 = d0*r0(h(i));
    d1 = d1*r1(h(i));
end
%d1>d0 return 1/d1<d0 return 0
d = sign (d1-d0);
end
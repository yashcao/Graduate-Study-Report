function [Q0,Q1] = CalcQ(h,f0,f1,r0,r1)
d0 = f0;
d1 = f1;
for i = 1:length(h)
    d0 = d0*r0(h(i));
    d1 = d1*r1(h(i));
end
Q0 = d0/(d0+d1);
Q1 = d1/(d0+d1);
end
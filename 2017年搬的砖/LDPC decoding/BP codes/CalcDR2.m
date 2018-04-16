function dR = CalcDR2(h,dQ)
dR=1;
for i=1:length(h)
    dR=dR*dQ(h(i));
end
end
function [dR] = CalcDR(h,dQ)
dR = zeros(1,length(dQ));
for i=1:length(h)
   [ dR(h(i))]= CalcDR2([h(1:i-1),h(i+1:end)],dQ);
end
end
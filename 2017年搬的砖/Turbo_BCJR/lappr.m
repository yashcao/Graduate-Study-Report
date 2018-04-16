% Calculates LAPPR at each stage using ALPHA, BETA and GAMM probabilities. 
% Returns P(X(k)=0), P(X(k)=1) and LAPPR(k) for all k's

function [p_x0,p_x1,lappr_1]=lappr(ALPHA,BETA,GAMMA,N)
    p_x0=zeros(1,N);
    p_x1=zeros(1,N);
    lappr_1=zeros(1,N);
    for i=1:N
        % P(xi)=sum(ALPHA(m')*GAMMA(m',m)*BETA(m))
        % ...续行；下一行code和上一行是连着的
        p_x1(i)=(ALPHA(1,i)*GAMMA(1,2*i)*BETA(2,i))+(ALPHA(2,i)*GAMMA(2,2*i)*BETA(4,i))+...
            (ALPHA(3,i)*GAMMA(3,2*i)*BETA(1,i))+(ALPHA(4,i)*GAMMA(4,2*i)*BETA(3,i));
        
        p_x0(i)=(ALPHA(1,i)*GAMMA(1,2*i-1)*BETA(1,i))+(ALPHA(2,i)*GAMMA(2,2*i-1)*BETA(3,i))+...
            (ALPHA(3,i)*GAMMA(3,2*i-1)*BETA(2,i))+(ALPHA(4,i)*GAMMA(4,2*i-1)*BETA(4,i));
        % lappr:后验概率的对数似然比
        lappr_1(i)=log(p_x1(i)/p_x0(i));      
    end
end
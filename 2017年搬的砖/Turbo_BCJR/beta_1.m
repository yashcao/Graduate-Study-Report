% BETA_k(m)=sum(GAMMA_k+1(m',m)*BETA_k+1(m')) 
% 前向计算使得ALPHA很小逼近0,为避免ALPHA=0
% multiply each ALPHA by 10^(20) at a stage where they become less than 10^(-20).

function [BETA]=beta_1(GAMMA,N)
    % Each column is for states 00,10,01,11 respectively.
    BETA=zeros(4,N);
    % BETA_k(m)=P(R[k+1:n]|s_k=m) ==> BETA_n(00)=P[s_n=00]=1 ==> s_n=00
    % Initialization assuming the final stage to be 00
    BETA(1,N)=1;BETA(2,N)=0;BETA(3,N)=0;BETA(4,N)=0;
    j=2*N-1;     % BETA_k-1(m')=sum(GAMMA_k(m',m)*BETA_k(m))
    for i=N-1:-1:1
       BETA(1,i)=(GAMMA(1,j)*BETA(1,i+1))+(GAMMA(1,j+1)*BETA(2,i+1));
       BETA(2,i)=(GAMMA(2,j)*BETA(3,i+1))+(GAMMA(2,j+1)*BETA(4,i+1));
       BETA(3,i)=(GAMMA(3,j)*BETA(2,i+1))+(GAMMA(3,j+1)*BETA(1,i+1));
       BETA(4,i)=(GAMMA(4,j)*BETA(4,i+1))+(GAMMA(4,j+1)*BETA(3,i+1));
       j=j-2; 
       
       if (BETA(1,i)<10^(-20) && BETA(2,i)<10^(-20) &&...
               BETA(3,i)<10^(-20) && BETA(4,i)<10^(-20) )
           BETA(:,i)=10^(20)*BETA(:,i);         % Scaling beta if became very less      
       end
    end   
end
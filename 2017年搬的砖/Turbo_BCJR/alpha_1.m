% ALPHA_k(m)=sum(GAMMA_k-1(m',m)*ALPHA_k-1(m')) 
% 前向计算使得ALPHA很小逼近0,为避免ALPHA=0
% multiply each ALPHA by 10^(20) at a stage where they become less than 10^(-20).

function [ALPHA]=alpha_1(GAMMA,N)
    % Each column is for states 00,10,01,11 respectively.
    ALPHA=zeros(4,N);
    % ALPHA_k(m)=P(s_k=m|R[0:k-1]) ==> ALPHA_1(00)=P[s_1=00]=1 ==> s_1=00
    % Initialization of alpha assuming first state to be 00
    ALPHA(1,1)=1;ALPHA(2,1)=0;ALPHA(3,1)=0;ALPHA(4,1)=0;  
    j=1;    % ALPHA(m,k)=sum(GAMMA_k-1(m',m)*ALPHA(m',k-1))
    for i=2:N
        ALPHA(1,i)=((GAMMA(1,j)*ALPHA(1,i-1))+(GAMMA(3,j+1)*ALPHA(3,i-1)));
        ALPHA(2,i)=((GAMMA(3,j)*ALPHA(3,i-1))+(GAMMA(1,j+1)*ALPHA(1,i-1)));
        ALPHA(3,i)=((GAMMA(2,j)*ALPHA(2,i-1))+(GAMMA(4,j+1)*ALPHA(4,i-1)));
        ALPHA(4,i)=((GAMMA(4,j)*ALPHA(4,i-1))+(GAMMA(2,j+1)*ALPHA(2,i-1)));
        j=j+2;
        % 针对每个特定码字概率放大
        if (ALPHA(1,i)<10^(-20) && ALPHA(2,i)<10^(-20) &&...
                ALPHA(3,i)<10^(-20) && ALPHA(4,i)<10^(-20) )
            ALPHA(:,i)=10^(20)*ALPHA(:,i);   % Scaling Alpha if became very less  
        end    
    end
end
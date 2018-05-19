function [h_out,support] = SD_based(x,Phi,L,V)
% n = length(h_hat);
n = size(Phi,2);
x_temp = x;
for l = 1:L
    % [~,order] = max(abs(h_hat));
    y = Phi'*x_temp;
    [~, temp2] = sort(sum(abs(y).^2,2),'descend');
    order =temp2(1);
    
    % select(l,:) = order - (9-1)/2 : 1 :order + (9-1)/2;
    if mod(V,2) == 0
        select(l,:) = order - (V)/2 : 1 :order + (V-2)/2; % V=8
    elseif mod(V,2) == 1
        select(l,:) = order - (V-1)/2 : 1 :order + (V-1)/2; % V=9
    end
    for i = 1:length(select(l,:))
        if select(l,i) > n
           select(l,i) = select(l,i) - n;
        elseif select(l,i)<1
           select(l,i) = select(l,i) + n;
        end
    end
    Phi2 = Phi(:,select(l,:));
    h_hat2 = inv(Phi2'*Phi2)*Phi2'*x_temp; % LS
%     if l<2
%        h_hat2 = inv(Phi2'*Phi2+sigma2*eye(length(select(l,:))))*Phi2'*x_temp;
%     else
%        h_hat2 = inv(Phi2'*Phi2+(sigma2/0.1)*eye(length(select(l,:))))*Phi2'*x_temp;
%     end
    temp = zeros(n,1);
    temp(select(l,:)) = h_hat2;
    if l>=1
       x_temp = x_temp - Phi*temp;
       % h_hat = OMP_new(x_temp, Phi, 9, 9);
    end
end
% select_final = unique(reshape(select,L*9,1));
select_final = unique(reshape(select,L*V,1));

Phi_final = Phi(:,select_final);
est =  inv(Phi_final'*Phi_final)*Phi_final'*x;
% est =
% inv(Phi_final'*Phi_final+sigma2*eye(length(select_final)))*Phi_final'*x;

support = select_final;

h_out = zeros(n,1);
h_out(select_final) = est;

% select = sort(order(1:V));
% set = 1:n;
% set(select)=[];
% Phi1 = Phi(:,set);
% if order(2)<order(1)
%     theta = (1/n)*(order(1)-(n+1)/2)-1/(4*n);
%     temp = U.'*sqrt(1/n)*exp(1i*[0:n-1]*2*pi*theta)';
% else
%     theta = (1/n)*(order(1)-(n+1)/2)+1/(4*n);
%     temp = U.'*sqrt(1/n)*exp(1i*[0:n-1]*2*pi*theta)';
% end
% temp(select)=0;
% h_est = h_hat + temp;
% x1 = x - Phi1*h_est(set);
% for i = 1:V
%     h_out(select(i)) = h_hat2(i);
% end

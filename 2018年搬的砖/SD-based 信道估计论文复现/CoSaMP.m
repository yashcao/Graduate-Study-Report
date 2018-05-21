function [a, support, used_iter] = CoSaMP(z,Phi,remaind_max_iter)

% Begin CoSaMP
err = 10^(-3);  % residual error

m = size(Phi,2); % 
n = size(z,2); % m x n 的稀疏矩阵
a = zeros(m, n); % 初始的稀疏矩阵 256x1

v = z - Phi * a;
it=0;
stop = 0;

while ~stop
    y = Phi'*v;
    % y = y./repmat( sqrt(sum(abs(Phi).^2,1)'/M),1,n); % 归一化
    % [temp1, temp2] = sort(sum(abs(y).^2,2),'descend');
    % [~, temp] = sort(sum(abs(y).^2,2),'descend');
    [~, temp] = sort(abs(y),'descend');
    Omega =temp(1:2*remaind_max_iter);
    
    if it==0
        T = Omega;
    else
        T = union(Omega, T_last); % 从小到大排序
    end
    % T_last = T;
    %Signal Estimation
    b = zeros(m, n);
    T_index = T;
    
    %Prune
    Phi2 = Phi(:,T_index);
    b(T_index,:) = inv(Phi2'*Phi2)*Phi2'*z; % LS
    %b(T_index,:) = Phi(:, T_index) \ z; % 类似于LS，得到保留位置抽头
    %b(T_index,:) = pinv(Phi2)*z; % LS

    % a = b; % 类似于LS，得到保留位置抽头，其他置为0
    
    % 在估计值中选绝对值最大的 k 列
    [~,temp2] = sort(abs(b),'descend');
    %[~, temp2] = sort(sum(abs(b).^2,2),'descend');
    T_index2 =temp2(1:remaind_max_iter);

    
    % 保留问题
    a = zeros(m,n);
    a(T_index2,:) = b(T_index2,:); % 类似于LS，保留位置抽头，其他置为0
    T_last = T_index2;
    
    %Sample Update
    v = z - Phi*a;
    %Iteration counter
    it = it + 1;
    
    %Check Halting Condition
    if (it >= max(remaind_max_iter) ||  norm(v)<=err*norm(z))   % norm(r_n)<=err  % Normally: max_iter = 6*(s+1)
      stop = 1;
    end
end
 
% 发射天线
support = unique(T_last);
used_iter = it;

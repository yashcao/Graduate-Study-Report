% FUNCTION 8.10 : "cp0802_PDP"
%
% Evaluates the Power Delay Profile 'PDP'
% of a channel impulse response 'h' sampled
% at frequency 'fc'
%
% Programmed by Guerino Giancola

function [PDP] = PDP(h,fc)

% --------------------------------
% Step One - Evaluation of the PDP
% --------------------------------

dt = 1 / fc;        % sampling time
PDP = (abs(h).^2)./dt;   % PDP

% ----------------------------
% Step Two - Graphical Output
% ----------------------------

Tmax = dt*length(h);
time = (0:dt:Tmax-dt);
    
S1=plot(time,PDP);
AX=gca;
set(AX,'FontSize',14);
T=title('Power Delay Profile');
set(T,'FontSize',14);
x=xlabel('Time [s]');
set(x,'FontSize',14);
y=ylabel('Power [V^2]');
set(y,'FontSize',14);
end

clear;

% ------------data------------
[bitsdata,RXdata] = gendata;

label = reshape(bitsdata,1,[]);
data = reshape(RXdata',75,[]);
% train_data = train_data';

train_label = label(:,1:1800);
test_label = label(:,1801:3000);
train_data = data(:,1:1800);
test_data = data(:,1801:3000);

% train_label = train_label';
% test_label = test_label';
% train_data = train_data';
% test_data = test_data';


pn=train_data; tn=train_label;
%[pn,input_str]=mapminmax(pn,0,1);%输入数据归一化
%[tn,output_str]=mapminmax(tn,0,1);%输出数据归一化
%net = feedforwordnet();
net=newff(pn,tn,[45,8],{'tansig','tansig' },'traingd');
net.trainParam.epochs=4000;%最大迭代次数
net.trainParam.show = 20 ; 
net.trainParam.goal=1e-2; 
net.divideFcn='';%如果超过六次迭代没有变化就会停止，用这个命令取消。
%net.trainFcn = 'trainrp'; % RPROP(弹性BP)算法,内存需求最小
net.trainParam.lr=0.2;%学习速度

% net.layers{1}.initFcn = 'initnw';
% net.layers{2}.initFcn = 'initnw';
% net.inputWeights{1,:}.initFcn = 'rands';
% net.inputWeights{2,:}.initFcn = 'rands';
% net.biases{1,:}.initFcn = 'rands';
% net.biases{2,:}.initFcn = 'rands';
net=init(net);%初始化网络 

net=train(net,pn,tn);%开始训练
an=sim(net,pn);
%a=mapminmax('reverse',an,output_str);
%plot(x,a);%画出当前图形（自变量和因变量）

pnew = test_data;
%pnew=mapminmax('apply',pnew ,input_str);%把准备预测的数据归一化（就是自变量）
anew=sim(net,pnew);
%anew=mapminmax('reverse',anew,output_str);%把预测的数据还原会原数量级

prelen = length(test_label);

hitNum = 0 ; anew2 = anew;
for i = 1 : prelen
    if anew(i)>0.5
        anew2(i) = 1;
    else
        anew2(i) = 0;
    end
    if( anew2(i) == test_label(i)   ) 
        hitNum = hitNum + 1 ; 
    end
end
sprintf('译码正确率是 %3.3f%%',100 * hitNum / prelen )






#  二维 DOA 估计算法

在实际通信中，用户位于三维空间，因此用户的 DOA 需要由二维角度来刻画。如果只是估计出用户的一维 DOA，那只是可以确定用户存在于一个平面，还不能精确的定向。因此利用二维 DOA 比只利用一维 DOA **方位角信息**能更有效的开发信道的冗余信息，更好的描述信号特征。     

二维 DOA 估计算法可以分为两大类：

- 利用**天线阵列**来完成空间二维DOA估计；

- 利用**空时二维处理**的方法来完成空间二维DOA估计。

## 空间阵列二维 DOA 估计

利用空间天线阵列来完成空间二维 DOA 估计的算法有两大类：

- 基于 MUSIC 方法的二维谱峰搜索方法，该算法需要**很大的计算量**。

- 基于 ESPRIT 类算法。该类算法利用特征值直接求解来波方向，相对于 MUSIC 类算法，在一定程度上降低了计算复杂度。但是利用 ESPRIT 类算法进行二维 DOA 估计时，由于常常是将其进行分维处理，所以存在**空间方位角和俯仰角参数配对问题**。

而常用的空间天线阵列模型有：平行线阵和 L 型天线阵列。 基于 ESPRIT 类算法，下面提出一种解决参数配对问题的二维 DOA 估计算法，其不仅解决了空间方位角和俯仰角参数配对问题，而且相对于其他二维来波方向估计算法，具有较少的计算复杂度。该二维 DOA 估计算法是基于平行天线的阵列模型。 

### 快速二维 DOA 估计 

系统模型如图所示，天线阵列为**双平行天线阵列**结构。$x$ 轴阵元间距为 $d_x$，两平行天线阵列间距为 $d_y$ 。第一个均匀线阵由 $(N +1)$ 个阵元组成，第二个均匀线阵由 $N$ 个阵元组成。假设位于远场的具有相同波长的 $P$ 个统计独立的窄带信号从二维方向 $(r_k,\varphi_k)$ 入射到阵列。     

![2d](E:\Documents\Markdown\2d.png)

其中 $(r_k,\varphi_k)$ 分别为**方位角**和**俯仰角**。方位角表示在 $xoy$ 面上的投影与 $x$ 轴夹角，俯仰角为投影与入射信号之间的夹角。   

![ae](E:\Documents\Markdown\ae.png)

为处理方便，下面用 $ (\alpha_k,\beta_k)$ 表示 $(r_k,\varphi_k)$，其中对应关系有 
$$
\begin{align*}
\cos \alpha_k &= \cos \varphi_k \cdot \cos r_k  \\
\cos \beta_k &= \cos \varphi_k \cdot \sin r_k  
\end{align*}
$$
从双平行线阵列结构构造 3 个子阵，其中第一个子阵为第一个均匀线阵的前 $N$ 个阵元组成，第二个子阵为第一个均匀线阵的后 $N$ 个阵元组成，第 3 个子阵为第二个均匀线阵 $N$ 个阵元组成。分别用 $X,Y,Z$ 表示子阵 1，2，3 接收的来波信号矩阵。子阵 1，2，3 的输出噪声为 $n_1,n_2,n_3$，假设它们为零均值且方差为 $\sigma^2$ 的高斯白噪声，且与信号矩阵不相关。需要注意的是，$n_1$ 和 $n_2$ 相关，它们和 $n_3$ 不相关。

$$
\begin{align*}
n_1 &= \left[ n_{11}(t),n_{12}(t),\cdots,n_{1N}(t)\right]^T \\
n_2 &= \left[ n_{21}(t),n_{22}(t),\cdots,n_{2N}(t)\right]^T \\
    &= \color{#F00}\left[ n_{12}(t),n_{13}(t),\cdots,n_{1,N+1}(t)\right]^T \\
n_3 &= \left[ n_{31}(t),n_{32}(t),\cdots,n_{3N}(t)\right]^T
\end{align*}
$$

假设来波信号 $s_i(t),\ (i=1,2,\cdots,P)$ 与 $x$ 轴夹角为 $\alpha_i$，与 $y$ 轴夹角为 $\beta_i$，则子阵所接收的来波信号矩阵为  
$$
\begin{align*}
X &= AS+n_1 \\
Y &= A \Phi_1 S+n_2 \\
Z &= A \Phi_2 S+n_3 \\
\end{align*}
$$
其中，$X = \left[ x_1(t),x_2(t),\cdots,x_N(t)\right]^T$，$Y,Z$ 类似。注意两个阵元之间的波程差如下图，同一个信号到达相邻阵元存在一个波程差 $\tau = \frac{d \sin \theta}{c}$，波程差导致了相位差 $\psi = e^{-jw\tau}$。

![d](E:\Documents\Markdown\d.png)
$$
\psi = e^{-jw\tau}=e^{-jw\frac{d \sin \theta}{c}}=e^{-j(2\pi f)\frac{d \sin \theta}{\lambda f}} = e^{-j2\pi \frac{d \sin \theta}{\lambda}}
$$

于是根据下图在一条线上的阵列几何关系，有 $c\tau = d_y\cos \beta$ 成立。

![jihe](E:\Documents\Markdown\jihe.png)

再考虑以原点为参考点，另外一个阵元位于 $(d_x,d_y)$，则有关系 $c\tau = d_x\cos \alpha+d_y \cos \beta$ 成立。

![jihe1](E:\Documents\Markdown\jihe1.png)

其中，$a(\alpha_i) = [1,e^{-j2\pi d \sin \alpha_i/ \lambda},\cdots,e^{-j(M-1)2\pi d \sin \alpha_i/ \lambda}]^T$。因此有 

$$
\begin{align*}
S &= \left[ s_1(t),s_2(t),\cdots,s_P(t)\right]^T \\
A &= \left[ a(\alpha_1),a(\alpha_2),\cdots,a(\alpha_P)\right] \\
\Phi_1 &= \text{diag} \left( \exp(j2\pi d_x \cos \alpha_1/ \lambda),\cdots,\exp(j2\pi d_x \cos \alpha_P/ \lambda)\right)^T \\
\Phi_2 &= \text{diag} \left( \exp(j2\pi d_y \cos \beta_1/ \lambda),\cdots,\exp(j2\pi d_y \cos \beta_P/ \lambda)\right)^T 
\end{align*}
$$

利用上述数据模型，可以得到下列数据矩阵

$$
\begin{align*}
R_{XX} &= \mathbb{E}(XX^H) = AR_SA^H+\sigma^2I \\
R_{XY} &= \mathbb{E}(XY^H) = AR_S\Phi_1^HA^H+\sigma^2Q \\
       &= AR_S\Phi_1^HA^H+\sigma^2\begin{bmatrix}
       \mathbf 0 & \mathbf 0 \\
       I_{N-1}   & \mathbf 0
       \end{bmatrix}\\
R_{XZ} &= \mathbb{E}(XZ^H) = AR_S\Phi_2^HA^H \\
R_{YY} &= \mathbb{E}(YY^H) = A\Phi_1R_S\Phi_1^HA^H+\sigma^2I \\
R_{YZ} &= \mathbb{E}(YZ^H) = A\Phi_1R_S\Phi_2^HA^H \\
R_{ZZ} &= \mathbb{E}(ZZ^H) = A\Phi_2R_S\Phi_2^HA^H+\sigma^2I
\end{align*}
$$

计算数据矩阵 $R_{XX}$ 的特征分解，由特征值大小排列估计噪声功率 $\sigma^2$。接下来我们令

$$
\begin{align*}
C_{XX} &= R_{XX}-\sigma^2I = AR_SA^H \\
C_{XY} &= R_{XY}-\sigma^2Q = AR_S\Phi_1^HA^H \\
C_{XZ} &= R_{XZ} = AR_S\Phi_2^HA^H \\
C_{YY} &= R_{YY}-\sigma^2I = A\Phi_1R_S\Phi_1^HA^H \\
C_{YZ} &= R_{YZ} = A\Phi_1R_S\Phi_2^HA^H \\
C_{ZZ} &= R_{ZZ}-\sigma^2I = A\Phi_2R_S\Phi_2^HA^H
\end{align*}
$$

利用上面的形式构造矩阵 $C$，得到  

$$
\begin{align*}
C &=    \begin{bmatrix}
       C_{XX} & C_{XY} & C_{XZ}\\
       C_{XY}^H & C_{YY} & C_{YZ} \\
       C_{XZ}^H & C_{YZ}^H & C_{ZZ} 
       \end{bmatrix} \\
  &=   \begin{bmatrix}
       AR_SA^H & AR_S\Phi_1^HA^H & AR_S\Phi_2^HA^H\\
       A\Phi_1R_SA^H & A\Phi_1R_S\Phi_1^HA^H & A\Phi_1R_S\Phi_2^HA^H \\
       A\Phi_2R_SA^H & A\Phi_2R_S\Phi_1^HA^H & A\Phi_2R_S\Phi_2^HA^H
       \end{bmatrix} \\
  & =  \begin{bmatrix}
       A \\
       A\Phi_1  \\
       A\Phi_2 
       \end{bmatrix}  R_S
       \begin{bmatrix}
       A^H &  \Phi_1^HA^H  & \Phi_2^HA^H 
       \end{bmatrix} \\
  & =  BR_SB^H  
\end{align*}
$$

如果 $P$ 个信号源来波方向 $\alpha,\beta$ 不同时接近，则有：$\text{Rank}(B) = P$。则可以对矩阵 $C$ 进行特征值分解得到子空间 $U_S$ ，显然信号子空间 $U_S$ 和 $B$ 张成的子空间一致，因此存在唯一的可逆矩阵 $T$ ，满足下面的关系：     

$$
U_S = \begin{bmatrix}
       U_1 \\
       U_2  \\
       U_3 
       \end{bmatrix}
       =\begin{bmatrix}
       A \\
       A\Phi_1  \\
       A\Phi_2 
       \end{bmatrix}T =BT
$$

根据以上关系可知，$U_2 = U_1T^{-1}\Phi_1T=U_1\Psi_1$，$U_3 = U_1T^{-1}\Phi_2T=U_1\Psi_2$。我们定义
$$
\begin{align*}
\Psi_1 &= T^{-1}\Phi_1 T = U_1^\dagger U_2 \\
\Psi_2 &= T^{-1}\Phi_2 T = U_1^\dagger U_3
\end{align*}
$$
其中，$U_1^\dagger = (U_1^HU_1)^{-1}U_1^H$。显然对 $\Psi_1,\Psi_2$进行特征分解可得到 $\Phi_1,\Phi_2$ 的特征值，就可得到入射角度 $(\alpha_i,\beta_i)$。由于 $\Psi_1,\Psi_2$ 单独处理，所以存在角度配对问题。我们做以下处理 
$$
\begin{align*}
\Psi_1 &= T_1^{-1}\Phi_1 T_1  \\
\Psi_2 &= T_2^{-1}\Phi_2 T_2 
\end{align*}
$$
这里 $T_1,T_2$ 是由矩阵 $T$ 的列组成的。通过比较特征向量矩阵 $T_1,T_2$ 的列向量可以决定二维 DOA 估计 $(\alpha_i,\beta_i)$ 的参数配对问题。在实际计算中 2 个特征值分解是独立进行的，可能特征向量的排列顺序是不同的，但同一信号的特征向量是**强相关**的。所以可构造排序矩阵 $G$，对特征值的顺序进行调整以实现参数配对。计算 
$$
G=T_2^HT_1
$$
根据 $G$ 中每一列元素中绝对值最大的一个元素的矩阵坐标来调整 $\Phi$ 中对角元素的顺序,实现特征向量的对应关系。进而可以求取信号的二维到达角。 

### matlab 仿真

```matlab
clear variables;
close all; clc;

degrad=pi/180;
radeg=180/pi;

N=5;     % 子阵阵元数
P=3;     % 信号数
snr=10;  % 信噪比 dB
dd=0.5;  % 元间距 0.5波长
k=1024;  % 快拍数

d=0:dd:(N-1)*dd;% dx=dy=dd
a = [30 50 70]; % x 轴夹角
b = [20 40 60]; % y 轴夹角
figure;
plot(a,b,'r*');
hold on;
twpi=2*pi;

A1 = [exp(-1i*twpi*d.'*(sin(a*degrad)))];
ph1= [exp(1i*twpi*dd.'*(cos(a*degrad)))];
ph2= [exp(1i*twpi*dd.'*(cos(b*degrad)))];
A2 = A1*diag(ph1);
A3 = A1*diag(ph2);

N0 = sqrt(1/2)*(randn(N+1,k)+1i*randn(N+1,k));
N1 = N0(1:N,:);
N2 = N0(2:N+1,:);
N3 = sqrt(1/2)*(randn(N,k)+1i*randn(N,k));

S = randn(P,k);
X = A1*S+N1;
Y = A2*S+N2;
Z = A3*S+N3;

Rxx=X*X'/k;
Rxy=X*Y'/k;
Rxz=X*Z'/k;
Ryy=Y*Y'/k;
Ryz=Y*Z'/k;
Rzz=Z*Z'/k;

% 估计噪声功率
[V,D]=eig(Rxx);   % 特征值 对角阵D和特征向量V
D=diag(D)';       % 把特征值对角阵变换成行矩阵
[D,~]=sort(D);    % 特征值V按从小到大排列，I为对应特征值的位置
sigma2 = mean(D(1:N-P)); 

Cxx=Rxx-sigma2*eye(size(Rxx));
Cxy=Rxy-sigma2*diag([ones(1,N-1)],-1);
Cxz=Rxz;
Cyy=Ryy-sigma2*eye(size(Ryy));
Cyz=Ryz;
Czz=Rzz-sigma2*eye(size(Rzz));

% 构造 C
C = [Cxx, Cxy, Cxz;
     Cxy',Cyy, Cyz;
     Cxz',Cyz',Czz];
% 估计噪声功率
[Ug,Sg]=eig(C);     % 特征值分解
Sg=diag(Sg)';       % 行向量

[Sg,I]=sort(Sg);    % 特征值V按从小到大排列，I为对应特征值的位置
Sg=fliplr(Sg);      % 特征值V按从大到小排列
Ug=fliplr(Ug(:,I)); % 特征矢量的列，对应相应的特征值，按列I翻转，变成由大到小排列
U=Ug(:,1:P);        % 信号子空间

U1 = U(1:N,:);
U2 = U(N+1:2*N,:);
U3 = U(2*N+1:3*N,:);

% Psi1 = inv(U1'*U1)*U1'*U2;
% Psi2 = inv(U1'*U1)*U1'*U3;
Psi1 = (U1'*U1)\U1'*U2;
Psi2 = (U1'*U1)\U1'*U3;
% x=uvu or x=(-u)v(-u) 因此后面求角度要加 abs
[T1,S1]=eig(Psi1);   % 特征值分解
[T2,S2]=eig(Psi2);   % 特征值分解

% 配对
% 特征值对应，则特征向量顺序一致，相乘应为单位阵
G=T2'*T1;      
G = abs(G);
[max_g,index]=max(G,[],2);
[Row,Col]=size(G);
for i=1:Row
    for j=1:Col
        if j == index(i)
            G(i,j)=1;
        else
            G(i,j)=0;
        end
    end
end

T1=T1*inv(G);
S1=inv(T1)*Psi1*T1;

S1=diag(S1)';    % 把特征值对角阵变换成行矩阵
S2=diag(S2)';    % 把特征值对角阵变换成行矩阵

an = acos(abs(angle(S1))/(twpi*dd))*radeg;
bn = acos(abs(angle(S2))/(twpi*dd))*radeg;

% an = acos(angle(S1)/(twpi*dd))*radeg
% bn = acos(angle(S2)/(twpi*dd))*radeg

plot(an,bn,'.')
axis([10 90 10 90]);
set(gca, 'XTick',[10:10:90]);
grid on;
```

仿真估计结果如图所示：

![test100](E:\Documents\Markdown\test100.png)

## 空时二维 DOA 估计

一般的二维  DOA 特征结构方法都会遇到二维谱峰搜索、非线性优化、分维处理及**参数配对**等难题，而且二维  DOA  估计**必须借助平面阵或立体阵，阵元数较多。因此计算量和存贮量变大**，降低了算法的实用性。另外，还需要通过**特殊的阵列结构**才能实现二维  DOA  估计，增加了阵列孔径损失，阵元利用率降低，阵元间互耦加剧，设备的成本、体积和复杂性增加，阵元的校正也十分困难。      

因此，在利用特征结构法进行空域处理的同时引入适当的时域处理，更加充分地利用信号中的信息。这样可以降低对阵列结构的约束，提高算法的抗噪能力。**将同时利用信号空域信息，以及时域信息的方法称为空时二维 DOA 估计法。**该算法**不需要双平行线阵**，也不存在角度兼并等问题，对噪声不敏感，并可推广到任意形状阵列且阵元大部分未校准的二维  DOA  估计问题。 

另外，在无线通信信道中，同信道干扰和码间干扰是限制天线容量和数据率的主要因素，利用时域一维处理或者利用空域一维处理都存在很多局限性，因此选用空时二维处理的方法则能取到较好的效果。典型的空时接收器是接收调制解调器，它同时对所有天线工作，并在时间和空间的两个域处理接收的信号。下面首先对空时二维处理方法数据进行系统建模。

![sm](E:\Documents\Markdown\sm.png)

上图所示均匀线阵模式，在 $M$ 个传感器的输出端加上 $L$ 个**延迟单元**，每个延迟单元的时延为 $\tau$。其中 $l$ 为延迟单元的序号，当 $l=0$ 时表示第 $m$ 个阵元的信号。则第 $m $个阵元连同其后的第 $l$ 个延迟单元上的信号可以表示为   
$$
x_{ml}(t) = \sum_{i=1}^{P}s_i\left(t+\frac{(m-1)d\sin \theta_i}{c}-l\tau \right)+n_{ml}(t)
$$
考虑所有的阵元，写成矩阵的形式为
$$
X_l = A_lS+n_l
$$
其中 $X_l$ 为第 $l$ 个延迟单元上的信号矩阵，$n_l$ 为零均值、方差为 $\sigma^2$ 统计独立的高斯白噪声，且与信号矩阵不相关。其中 $A_l = A_0 \Phi_l$。 

当不存在延迟时，有
$$
X_0 =A_0S+n_0 
$$
这里 $A_0=\left[ a(\theta_1),\cdots, a(\theta_p)\right]$，其中 $a(\theta_p) = \left[ 1,e^{-jw(d\sin \theta_p/c)},\cdots,e^{-jw((M-1)d\sin \theta_p/c)}\right]^T$。

可知，通过**时域的延时**，可以得到**虚拟**阵元形式，使得算法**能处理的入射信源个数大于等于阵元个数**。由于实际延迟单元的延迟时间可知，所以可以结合时间域上所得到虚拟天线阵列和空间域上的天线阵列进行空时二维 DOA 估计。

### 二维虚拟 ESPRIT 算法

数据模型如下图所示的天线接收阵列，它由三个子线性阵列组成。每个子阵列有 $N＋1$ 个阵元，子阵列相互平行，且完全相同，子阵列里每个阵元的间距均为 $d_x$ ，子阵列之间的距离为 $d_y$。  

![2S](E:\Documents\Markdown\2S.png)

定义 $Y_0$ 为 $1～N$ 阵元输出序列，$Y_1$ 和 $Y_2$ 为 $2～(N+1)$ 阵元输出序列。信号为窄带远场平面波时，假设来波信号 $s_i(t),(i=1,2,\cdots,p)$ 与 $x$ 轴夹角为 $\alpha_i$ ，与 $y$ 轴夹角为 $\beta_i$。子阵 $Y_0,Y_1,Y_2$ 的输出噪声为 $n_0,n_1,n_2$，假设它们为零均值的方差为 $\sigma^2$ 统计独立的高斯白噪声，且与信号矩阵不相关。则有  
$$
\begin{align*}
Y_0 &= AS+n_0 \\
Y_1 &= A\Phi_1S+n_1 \\
Y_2 &= A\Phi_2S+n_2 \\
Y_0 &= \left[ y_{01}(t),y_{01}(t),\cdots,y_{0N}(t)\right]^T \\
Y_1 &= \left[ y_{11}(t),y_{11}(t),\cdots,y_{1N}(t)\right]^T \\
Y_2 &= \left[ y_{21}(t),y_{21}(t),\cdots,y_{2N}(t)\right]^T \\
S   &= \left[ s_{1}(t),s_{2}(t),\cdots,s_{p}(t)\right]^T \\
A   &= \left[ a(\alpha_1),a(\alpha_2),\cdots,a(\alpha_p)\right] 
\end{align*}
$$
由上面定义可以看到，$Y_1,Y_2$ 并不是简单的平行于 $Y_0$ ，而是滞后 $Y_0$ 于一个子阵列的间隔单元 $d_x$。则有 
$$
\begin{align*}
\Phi_1 &= \text{diag}\left( e^{j(u_1+v_1)},e^{j(u_2+v_2)},\cdots,e^{j(u_p+v_p)}\right)\\
\Phi_2 &= \text{diag}\left( 
e^{j(u_1-v_1)},e^{j(u_2-v_2)},\cdots,e^{j(u_p-v_p)}\right)\\
u_i &= \frac{2\pi d_x \cos{\alpha_i}}{\lambda} \\
v_i &= \frac{2\pi d_y \cos{\beta_i}}{\lambda} 
\end{align*}
$$
利用上述数据模型，令
$$
\begin{align*}
Y &= \frac{Y_1+Y_2}{2} \\
  &= \frac{A(\Phi_1+\Phi_2)S+(n_1+n_2)}{2} \\
  &= A\Phi S+\frac{(n_1+n_2)}{2} \\
\Longrightarrow Y &= A\Phi S+N_2
\end{align*}
$$
复指数 $e^{jx}=\cos x+j \sin x$。利用和差化积公式
$$
\begin{align*}
\sin x + \sin y &= 2\sin\frac{x+y}{2}\cos\frac{x-y}{2} \\
\cos x + \cos y &= 2\cos\frac{x+y}{2}\cos\frac{x-y}{2} \\
\frac{1}{2} (e^{jx}+e^{jy}) &= \cos\frac{x-y}{2}\left[\cos\frac{x+y}{2}+j\sin\frac{x+y}{2} \right] \\
&= \cos\frac{x-y}{2}\exp\left(j\frac{x+y}{2}\right)
\end{align*}
$$
可得，$ \Phi = \text{diag}\left( \cos v_1e^{ju_1},\cos v_2e^{ju_2},\cdots,\cos v_pe^{ju_p}\right)$。 

由 $\Phi$ 的形式可知，利用以前的 ESPRIT 算法，可以求出 $\Phi$ 的模值和相位，就能求出相应的方位角和俯仰角，故能达到计算量小、估计精度高、对子阵列要求不高的优点。为了叙述方便，令 
$$
X = Y_0 = AS+n_0 \Longrightarrow X = AS+N_1
$$
于是可得
$$
\begin{align*}
R_{XX} &= \mathbb{E}\left( X(n)X^H(n)\right) = AR_SA^H+\sigma^2I \\
R_{XY} &= \mathbb{E}\left( X(n)Y^H(n)\right) = AR_S\Phi^HA^H \\
\Longrightarrow \\
C_{XX} &= R_{XX}-\sigma^2I = AR_SA^H \\
C_{XY} &= R_{XY} = AR_S\Phi^HA^H \\
\Longrightarrow \\
C = C_{XX} - \gamma C_{XY} &= AR_SA^H-AR_S(\gamma\Phi^H)A^H \\
&= AR_S(I-\gamma\Phi^H)A^H
\end{align*}
$$

> 若矩阵 $A,B$，满足 $Au = \xi Bu$，则称 $\xi$ 是 $\{A,B\}$ 的**广义特征值**，$u$ 是广义特征向量。$(\xi,u)$ 是广义特征对。 
>
> 当且仅当 $\xi$ 是 $\{A,B\}$ 的广义特征值时，$A-\xi B$ 奇异。广义特征值是广义特征多项式 $\big\vert A-\xi B \big\vert=0$ 的根。

假设信号角度不相关，所以 $A,R_S$ 的秩都为 $p$，故奇异值为 $\big\vert I-\gamma\Phi^H \big\vert=0$ 的根。 注意在这里求多个 $\gamma$ 值组成的对角阵的对角元素，且 $\Phi^{\color{#F00}{H}} = \text{diag}\left( \cos v_1e^{-ju_1},\cdots,\cos v_pe^{-ju_p}\right)$。于是得到
$$
\gamma = \text{diag}\left( \frac{1}{\cos v_1}e^{ju_1},\cdots,\frac{1}{\cos v_p}e^{ju_p}\right)
$$
到目前为止，建模为

$$
\begin{align*}
X &= AS+N_1 \\
Y &= A\Phi S+N_2 \\
\end{align*}
$$
通过前面的分析，可以知道我们只要获得 $\Phi$ 或者 $\gamma$，就可以求出角度。一种方法是利用求 $\{C_{XX},C_{XY}\}$ 的广义特征值得到 $\gamma$。另一种方法是按照 TLS-ESPRIT 的思路求 $\Phi$，可以进行如下操作： 
$$
\begin{align*}
Z &=\begin{bmatrix}
X \\ Y 
\end{bmatrix}
=\begin{bmatrix}
AS+N_1 \\ A\Phi S+N_2 
\end{bmatrix} \\
&=\begin{bmatrix}
A \\ A\Phi 
\end{bmatrix}S+\begin{bmatrix}
N_1 \\ N_2 
\end{bmatrix} \\
&=\bar{A}S+N
\end{align*}
$$
然后求 $R_{Z}=\mathbb{E}(ZZ^H)=\frac{1}{K} \sum_{t=1}^{K}Z(t)Z(t)^H$，进行分解得到信号子空间 $U_S$。根据
$$
U_S = \begin{bmatrix}
U_X \\U_Y
\end{bmatrix}
=\begin{bmatrix}
A \\ A\Phi 
\end{bmatrix}T
$$
记 $U_{ST} = \left[ U_X\vert U_Y\right]$，然后有  
$$
\begin{align*}
U_{ST}^HU_{ST} &= E\Sigma E^H \\
&= \begin{bmatrix} E_S & E_N
\end{bmatrix} 
\begin{bmatrix} \Lambda_S & \\ & \Lambda_N
\end{bmatrix}
\begin{bmatrix} E_S^H \\ E_N^H
\end{bmatrix} \\
\end{align*}
$$
最后得到
$$
\begin{align*}
E &= \begin{bmatrix} E_{11}& E_{12} \\ E_{21}& E_{22}
\end{bmatrix}  \\
E_N &= \begin{bmatrix} E_{12} \\ E_{22}
\end{bmatrix} \\
\Psi _{TLS} &= -E_{12} E_{22}^{-1}
\end{align*}
$$
通过 $\Psi _{TLS}$ 最终得到 $\Phi$。利用 $ \Phi = \text{diag}\left( \cos v_1e^{ju_1},\cos v_2e^{ju_2},\cdots,\cos v_pe^{ju_p}\right)$ 求出 $u$ 和 $v$。

```matlab
u = angle(phi);
cos_v = abs(phi);
v = acos(cos_v);
a = acos(lamda*u/(2*pi*dx));
b = acos(lamda*v/(2*pi*dy));
```

### 虚拟波达方向矩阵法

改进的二维虚拟 ESPRIT 算法：**虚拟波达方向矩阵法**中，其数据模型与二维虚拟 ESPRIT 算法一致。根据二维虚拟 ESPRIT 算法，求出 $\Phi$ 即可。虚拟波达方向矩阵法与虚拟 ESPRIT 算法的不同就在于求 $\Phi$ 的过程不同。将波达矩阵法的求解思想用在求解 $\Phi$ 上，从而得到虚拟波达方向矩阵法。其相对于二维虚拟ESPRIT算法，减少了计算复杂度，更适合于低信噪比，多信源的通信环境。

系统模型为
$$
\begin{align*}
X &= AS+N_1 \\
  &=Y_0 = AS+n_0 \\
Y &= A\Phi S+N_2 \\
  &= A\Phi S+\frac{(n_1+n_2)}{2} 
\end{align*}
$$
协方差以及估计噪声功率等相关操作
$$
\begin{align*}
R_{XX} &= \mathbb{E}\left( X(n)X^H(n)\right) = AR_SA^H+\sigma^2I \\
R_{YX} &= \mathbb{E}\left( Y(n)X^H(n)\right) = A\Phi R_S A^H \\
\Longrightarrow \\
C_{XX} &= R_{XX}-\sigma^2I = AR_SA^H \\
       &= V\Sigma V^H
\end{align*}
$$
式中 $\Sigma = \left[\mu_1,\cdots,\mu_N \right]$ 是特征值，而 $\{V_1,\cdots,V_N\}$ 是相应的特征向量，由前面 MUSIC 算法的信号子空间和噪声子空间的分析可知：

- $C_{XX}$ 的特征值 $\mu_{1} \geqslant \mu_{2} \geqslant \cdots \geqslant \mu_{p} > \mu_{p+1}= \cdots = \lambda_{N}=0$。 
- 特征向量组 $V = [V_s,V_n]$。$\text{span}\{ V_{p+1},V_{p+2},\cdots,V_N\} \perp \text{span}\{ A\}$，因此 $A^HV_n = 0$。    

于是有
$$
\begin{align*}
A^HV_nV_n^HA &= 0 \\
A^HVV^HA &= A^H(V_sV_s^H)A+A^H(V_nV_n^H)A \\
         &= A^HV_sV_s^H A \\
         &= A^H A
\end{align*}
$$
有了上述性质，推出波达方向矩阵法，记波达方向矩阵 $R$：
$$
R = R_{YX}*C_{XX}^{\dagger}
$$
其中 $C_{XX}^{\dagger} =V\Sigma^{-1} V^H$ 。

> 有以下推导
> $$
> \begin{align*}
> C_{XX} &= AR_SA^H \\
> R_SA^H &= A^{\dagger}C_{XX} \\
>        &= (A^HA)^{-1}A^H C_{XX} \\
> R_{YX} &= A\Phi R_S A^H \\ 
>        &= A\Phi (A^HA)^{-1}A^H C_{XX} \\
>     RA &= R_{YX}*C_{XX}^{\dagger}A \\
>        &= A\Phi (A^HA)^{-1}A^H C_{XX} *C_{XX}^{\dagger}A \\
>        &= A\Phi
> \end{align*}
> $$
> 可得结论：$RA = A\Phi$。对波达方向矩阵 $R$ 进行特征分解得到 $\Phi$ ，根据空间方位夹角与 $\Phi$ 之间对应关系得出来波方向角 $(\alpha, \beta)$ 。 

虚拟波达方向矩阵法与二维虚拟 ESPRIT 算法的不同点在于，虚拟波达方向矩阵法只需要一次 $(N×N)$ 的互协方差运算，一次 $(N×N)$ 互协方差的求逆，以及一次 $(N×N)$ 的特征值分解。而二维虚拟 ESPRIT 算法需要一次 $(2N×2N)$ 阶，一次 $(2p×2p)$ 阶和一次 $(p×p)$ 阶的特征值分解，所以当信源数较大时使用虚拟波达方向矩阵法可以大大节省算法的运算量。
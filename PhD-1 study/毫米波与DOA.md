# 毫米波与 DOA 估计

## 1. 毫米波技术的背景

无线网络数据量的巨大需求，新型应用层出不穷，迫切需要 5G 技术解决**提高数据速率和降低时延**的问题。边缘速率即用户最低数据速率要从目前 4G 系统的 1Mbps 提升到 100Mbps 甚至 1Gbps，另外 5G 的峰值速率则需要达到 20Gbps。当前 4G 系统的往返延迟大约为 15ms，虽然这种延迟对于现在大多数应用是足够的，但 5G 的时延需要下降到 1ms。

为了能够达到高数据率，可以通过以下三种方式来实现：

- 使用更小的蜂窝小区，实现跨区域重用频谱，减少每个基站下竞争资源的用户数量。
- 使用毫米波技术增加带宽。
- 使用 MIMO 技术增加频谱效率。

在此背景下，用于室内和室外无线通信的毫米波技术已成为提供高速数据链路的新方法。毫米波用于室内通信的可行性已被证明。从一些标准的制定中也可以看出业界对毫米波应用前景的重视。除此之外，学术界、工业界和政府都对毫米波在 5G 蜂窝系统中的应用产生极大的兴趣。

虽然毫米波频段带来优势，但要想发展毫米波通信技术，信号处理技术就显得格外重要。因为信号处理在毫米波频段上与在低频段有所不同，主要的原因是：

- 毫米波的**信道模型**和特点与低频段不同
- 毫米波频率高、带宽大，使得硬件实现受到限制
- 毫米波系统中需要在**收发两端应用大量天线阵元**

自由空间传播损耗，根据 Friis 定理我们知道，发射功率 $P_t$ 和远场接收功率 $P_r$ 存在如下关系：
$$
P_r = G_r G_t \left( \frac{\lambda}{4\pi d}\right)^2 P_t
$$
其中，$d$ 表示收发间距，$\lambda$ 表示信号波长，$G_t$ 表示发射增益，$G_r$ 表示接收增益。可以看出，各向同性路径损耗即 $G_r= G_t=1$ 情况下的 $\frac{P_t}{P_r}$ 与 $\lambda^2$ 成反比。这也就意味着与较低频段相比，因为毫米波**波长更短**，所以毫米波频段有着**更高的路径损耗**。此外，毫米波的衍射能力降低，散射现象严重，**穿透损耗更大**。 

因为毫米波的路径损耗大，所以更适合在密集化的小区中使用，因为这些蜂窝小区半径都比较小 (< 200m)，毫米波通信由于毫米波波长小，单位面积的天线阵列可以部署更多天线阵元，因此毫米波更适合使用大规模天线阵列。使用大规模天线阵列不仅可以提高天线增益以弥补较大的路径损耗，还可以满足多流/多用户的需求。同时，在室外环境中毫米波的散射路径会因为路径损耗较大而变少，因此可根据接收信号不同的到达方向来区分不同信源，这使得 DOA 估计技术可以在毫米波大规模天线阵列中得到应用，并且还可以使用波束成形技术来增强天线阵列在某一方向上的发射和接收增益，并以此来解决路径损耗大的问题。

但是，因为毫米波频率高、带宽大，给硬件实现带来了挑战，原本在低频段中使用的 DOA 估计和波束成形技术无法直接搬到毫米波通信系统中，于是研究适用于毫米波的低复杂度 DOA 估计成为了大规模天线阵列应用到毫米波通信中的重要因素。 

## 2. DOA 发展现状

- MUSIC：谱峰搜索导致计算量大，非相干。
- ESPRIT：减小计算量，避免谱峰搜索。
- 加权子空间拟合 (Weight Subspace Fitting, WSF) 算法：估计精度更高，可以处理相干的信号源，算法复杂度极高，导致其运算时间过长。
- 非相干信号子空间 (Incoherent Signal Subspace, ISM) 算法：最早出现的**宽带信号 DOA 估计算法**，主要原理是将宽带信号分为若干个窄带分量，然后使用经典的窄带 DOA 估计技术进行谱估计。但是 ISM 算法最大缺陷是**不适用于相干的信号源**。 
- 相干信号子空间 (Coherent Signal Subspace, CSM) 算法：该方法比 ISM 算法的估计精度高，**适用于相干的信号源**。但是，由于需要给 CSM 算法一个 DOA 预估值，CSM 算法的分辨率受到该预估值的影响。 
- 近几年为了处理**非均匀噪声**和**阵列不确定性**，还提出了通过**最大化对数似然函数**，或**求解最小二乘最小化问题**来确定信号子空间和噪声协方差矩阵，不需要阵列流形的精确信息便可以直接计算信号子空间，然后再根据子空间来估计 DOA，由于信号子空间和噪声协方差矩阵是以迭代闭合形式计算，因此可以有效降低复杂度。同时因信号带宽不断变大，还提出了一种宽带稀疏频谱拟合估计算法，它有效利用了稀疏空间采样中增加的自由度，并且也可以在非均匀噪声环境中拥有良好性能。
- 根据角度域的稀疏性，提出了基于**贝叶斯压缩感知学习**的新型 DOA 估计算法，可以实现更精确的 DOA 估计。

由于毫米波信号波长较短，天线阵元之间的距离小，于是在毫米波移动设备中可以运用密集的天线阵列，不同的信道路径可以根据不同的到达角进行区分。所以，就可以使用 DOA 估计技术对到达角进行估计。**在毫米波通信中到达移动接收机的路径数很少，在非视距 NLOS 信道中平均只有 2.4 条**。因此压缩感知技术可以有效地用于稀疏空间的信道估计和 DOA 估计。 

随着 5G 毫米波通信技术发展，宽带信号传输能够大大提高系统传输速率。但是，**大带宽导致信号在天线阵列之间传输时间造成的相移不可忽略**，并且还需要考虑频域的非平稳特性。但现有信道参数估计算法都仅适用于系统传输窄带信号的情况，还都不能应对这一问题。在使用大规模阵列进行信号传输或接收时，存在空间上的非平稳特性以及沿着阵列轴的簇的生灭，同时也需要**考虑波阵面为球面波而不再是平面波**。在文献[^1]中将 Tx 到第一阶散射体以及 Rx 到最后一阶散射体之间的距离考虑在内，从而估计路径在球面波假设下的角度和时延。并且，以上问题的解决需要革新性的、能够估计球面波参数、处理时变参数和信道 “时-空-频” 非平稳特性的高分辨率算法。 

## 3. 天线阵列模型

天线阵元之间的摆放位置影响着阵列接收信号的数学模型，不同的天线阵列模型有着不一样的应用场景，并影响着阵列信号处理的方法。

### 3.1 均匀线阵

![ula1](E:\Documents\Markdown\ula1.png)

假设 $M$ 个阵元等距离排列成一条直线，阵元间距为 $d$。假设选取最左边的天线阵元作为参考点，则第 $m$ 个天线阵元相对于参考点的时间延迟可以表示为  
$$
\tau_m(\theta_i) = \frac{(m-1)d \sin \theta_i}{v}
$$

- 各天线阵元之间的互耦效应忽略不计，且阵元间距为最高频率源信号的半波长，即 $d=\frac{\lambda}{2}$。

- 传播距离远大于阵列大小，即信号在介质中以平面波的形式到达阵列。

- 接收机各个通道拥有完全相同的特性。 

根据以上假设，方向向量可以写为
$$
\mathbf{a}(\theta_i) = \left[1,e^{j\pi \sin \theta_i},e^{j2\pi \sin \theta_i},\cdots,e^{j(M-1)\pi \sin \theta_i} \right]^T
$$

### 3.2  L 型阵

![ll](E:\Documents\Markdown\ll.png)

如图为 L 型阵列模型，在 $x$ 轴和 $y$ 轴均 $M$ 个阵元，阵元间距为 $d$。 $K$ 个信源，二维入射角为 $(\theta_i,\varphi_i)$。$x$ 轴上的接收信号模型可以视为线性阵列，$y$ 轴同理： 
$$
\begin{align*}
X &= A_xS+N_x \\
Y &= A_yS+N_y
\end{align*}
$$


   其中，方向矩阵为
$$
\begin{align*}
A_x &= \left[a_x(\theta_1,\varphi_1),\cdots,a_x(\theta_K,\varphi_K)\right] \\
A_y &= \left[a_y(\theta_1,\varphi_1),\cdots,a_y(\theta_K,\varphi_K)\right] \\
a_x(\theta_i,\varphi_i) &= \left[1,e^{-j\frac{2\pi}{\lambda}d \cos\theta_i \sin\varphi_i},\cdots,e^{-j\frac{2\pi}{\lambda}(M-1)d \cos\theta_i \sin\varphi_i}\right]^T \\
a_y(\theta_i,\varphi_i) &= \left[1,e^{-j\frac{2\pi}{\lambda}d \sin\theta_i \sin\varphi_i},\cdots,e^{-j\frac{2\pi}{\lambda}(M-1)d \sin\theta_i \sin\varphi_i}\right]^T 
\end{align*}
$$

### 3.3  均匀平面阵

![mian](E:\Documents\Markdown\mian.png)

假设有 $K$ 个信号源，均匀平面阵列指的是 $M$ 个天线阵元等间距的排列成一个正方形或者矩形，如上图所示由 $M\times N$ 个阵元组成。$x$ 轴方向有 $N$ 个间距为 $d$ 的均匀线阵，$y$ 轴方向有 $M$ 个间距为 $d$ 的均匀线阵。如果选取原点为参考点，另外某个阵元的坐标可以写为 $(x_n,y_m)$，于是时延差可以写为 

$$
\begin{align*}
\tau_{n,m}(\varphi_k,\theta_k) &= \frac{x_n\sin\varphi_k\cos\theta_k + y_m\sin\varphi_k\sin\theta_k}{v}\\
a_{n,m}(\varphi_k,\theta_k) &= \exp\left(-jw_0\tau_{n,m}(\varphi_k,\theta_k)\right) \\
X &= AS+N  \\
S &= \left[s_1(t),\cdots,s_K(t)\right]^T \in \mathbb{C}^{K \times 1}\\
A &= \left[a(\theta_1,\varphi_1),\cdots,a(\theta_K,\varphi_K)\right] \in \mathbb{C}^{NM \times K}
\end{align*}
$$

 从 $x$ 轴对应的 $N$ 个阵元的方向矩阵和从 $y$ 轴对应的 $M$ 个阵元的方向矩阵为 
$$
\begin{align*}
A_x &= \left[a_x(\theta_1,\varphi_1),\cdots,a_x(\theta_K,\varphi_K)\right] \\
A_y &= \left[a_y(\theta_1,\varphi_1),\cdots,a_y(\theta_K,\varphi_K)\right] \\
a_x(\theta_i,\varphi_i) &= \left[1,e^{-j\frac{2\pi}{\lambda}d \cos\theta_i \sin\varphi_i},\cdots,e^{-j\frac{2\pi}{\lambda}(N-1)d \cos\theta_i \sin\varphi_i}\right]^T \\
a_y(\theta_i,\varphi_i) &= \left[1,e^{-j\frac{2\pi}{\lambda}d \sin\theta_i \sin\varphi_i},\cdots,e^{-j\frac{2\pi}{\lambda}(M-1)d \sin\theta_i \sin\varphi_i}\right]^T 
\end{align*}
$$
![gs](E:\Documents\Markdown\gs.png)

则可以得到方向矩阵
$$
\begin{align*}
X &= AS+N  \\
S &= \left[s_1(t),\cdots,s_K(t)\right]^T \in \mathbb{C}^{K \times 1}\\
A &= \left[a(\theta_1,\varphi_1),\cdots,a(\theta_K,\varphi_K)\right] \\
  &= \begin{bmatrix}
     A_x D_1(A_y) \\
     A_x D_2(A_y) \\
     \vdots \\
     A_x D_M(A_y) 
     \end{bmatrix}
\in \mathbb{C}^{NM \times K}
\end{align*}
$$
其中 $D_i(\cdot)$ 表示取矩阵的第 $i$ 行作为构成对角矩阵的对角元素。 

### 3.4  均匀圆阵

![yuan](E:\Documents\Markdown\yuan.png)

如图所示为均匀圆阵模型，$M$ 个阵元均匀分布在圆周上，假设 $K$ 个信源，二维入射角为 $(\theta_i,\varphi_i)$，一般取圆周上两个阵元的间距为 $\lambda/2$，对应的圆半径取为 $R = \frac{\lambda}{4} / \sin(\frac{\pi}M)$，阵列的第 $m$ 个阵元与 $x$ 轴的角度用 $\frac{2\pi}{M}\cdot (m-1)$ 表示。 

![yuan1](E:\Documents\Markdown\yuan1.png)

以原点为参考点，则位于 $x$ 轴正方向的阵元视为沿着半径方向为轴的参考系。则有时延差：  
$$
\tau_{m=1,i} = \frac{R \sin\varphi_i \cos\theta_i}{v}
$$
其中 $\theta_i$ 是入射投影与半径方向的轴之间的夹角。同理第 $m$ 个阵元即把阵元与原点之间的半径方向作为参考轴，此时入射投影与半径方向的轴之间的夹角为 $\theta_i-\frac{2\pi(m-1)}{M}$。则有时延差：  
$$
\tau_{m,i} = \frac{R \sin\varphi_i \cos\left(\theta_i-\frac{2\pi(m-1)}{M}\right)}{v}
$$
于是阵列方向向量为
$$
\begin{align*}
A &= \left[a(\theta_1,\varphi_1),\cdots,a(\theta_K,\varphi_K)\right] \\
a(\theta_i,\varphi_i)  &= \begin{bmatrix}
\exp\left(-j2\pi R\sin\varphi_i \cos\left(\theta_i\right)\right) \\
\exp\left(-j2\pi R\sin\varphi_i \cos\left(\theta_i-\frac{2\pi}{M}\right)\right)
\\
\vdots \\
\exp\left(-j2\pi R\sin\varphi_i \cos\left(\theta_i-\frac{2\pi(M-1)}{M}\right)\right)
     \end{bmatrix}
\end{align*}
$$

### 3.5 任意三维阵列天线模型

![rand](E:\Documents\Markdown\rand.png)

如图所示为任意阵列模型。假设 $M$ 个阵元任意分布在空间中，二维入射角为 $(\theta_i,\varphi_i)$，假设有 $K$ 个信号源。则入射的方向矢量为：    
$$
\mathbf{V} = \begin{bmatrix}
\sin\varphi_i\cos\theta_i & 
\sin\varphi_i\sin\theta_i &
\cos\varphi_i 
\end{bmatrix}^T
$$
如果第 $m$ 个阵元的坐标位置为 $\mathbf{r}_m = ( x_m, y_m, z_m)$ ，波速为 $v$。那么     
$$
\tau_{m}(\theta_i,\varphi_i) = \frac{1}{v} 
\left(
x_m \sin\varphi_i\cos\theta_i + y_m \sin\varphi_i\sin\theta_i
+ z_m \cos\varphi_i
\right)
$$
因此可得导向矢量：
$$
\mathbf{a}(\theta_i,\varphi_i) = \begin{bmatrix}
p_1(\theta_i,\varphi_i) \exp(j \frac{2\pi}{\lambda} \mathbf{r}_1 \cdot \mathbf{V}) \\ 
p_2(\theta_i,\varphi_i) \exp(j \frac{2\pi}{\lambda} \mathbf{r}_2 \cdot \mathbf{V}) \\ 
\vdots\\
p_M(\theta_i,\varphi_i) \exp(j \frac{2\pi}{\lambda} \mathbf{r}_M \cdot \mathbf{V})  
\end{bmatrix}
$$
对于传统阵列，通常极化矢量 $p_k(\theta,\varphi) =1$ 一般省略。但是对于极化敏感阵列，不同载体的影响产生屏蔽效应，所以 $p_k(\theta,\varphi)$ 不能省略。

定义传统阵列方向矩阵为：$A = \left[a(\theta_1,\varphi_1),\cdots,a(\theta_K,\varphi_K)\right]$。 

## 4. 宽带信号阵列模型

**因为窄带信号的时延信息可以通过相位的变化来体现，而宽带信号则不满足这一假设**，所以上面提到的MUSIC/ESPRIT 等算法不适用于宽带信号。现在比较经典的宽带 DOA 估计算法是 ISS 算法和 CSS 算法。 

在有 $M$ 个天线单元的均匀线阵中，第 $m$ 个阵元接收到的信号可以写为   
$$
x_m(t) = \sum_{i=1}^{K} s_i(t-\tau_m(\theta_i))+n_m(t),\quad m=1,2\cdots,M
$$
其中，$s_i(t)$ 为第 $i$ 个信源，$\tau_m(\theta_i)$ 为其相对于参考点的时延，$\theta_i$ 为其入射角度，$K$ 为信源数目，$n_m(t)$ 为高斯白噪声。**因为宽带信号的时延无法用相移来表示，所以需要转换到频域上**，表示为  
$$
X_m(f) = \sum_{i=1}^{K} S_i(f)e^{-j2\pi f\tau_m(\theta_i)}+N_m(f)
$$
其中 $S_i(f)$ 和 $N_m(f)$ 分别为 $s_i(t)$ 和 $n_m(t)$ 傅里叶变换。我们可以将阵列接收信号的频域模型写成矩阵的形式，令 $e^{-j2\pi f\tau_m(\theta_i)} = a_m(f,\theta_i)$，可得  
$$
X(f) = A(f,\theta)S(f)+N(f)
$$
其中有
$$
\begin{align*}
A(f) &= [a(f,\theta_1),a(f,\theta_2),\cdots,a(f,\theta_K)] \\
a(f,\theta_i) &= [a_1(f,\theta_i),a_2(f,\theta_i),\cdots,a_M(f,\theta_i)]^T
\end{align*}
$$

### 4.1 非相干信号子空间算法

ISM 算法的核心思想是把一个宽带信号通过 FFT 在频域分解成若干个窄带分量，然后在每一个子带上直接应用窄带 DOA 估计技术进行处理，最后对这若干个结果进行综合，比如所有子带的空间谱进行平均，即可得到最终的 DOA 信息。

根据这个思想，首先把观测时间 $T_0$ 内的接收信号分成 $L$ 段，$L$ 也称为频域快拍。再对每段作 DFT 分解为 $N$ 个窄带分量，即得到 $L​$ 组互不相关的窄带频域分量，因此可得**接收信号频域自相关矩阵**的估计值 
$$
R_x(f_n) = \frac{1}{L} \sum_{l=1}^{L}X_l(f_n)X_l^H(f_n),\quad  1 \leq n \leq N
$$
其中 $X_l(f_n) = A(f_n,\theta)S(f_n)+N(f_n) \in \mathbb{C}^{M\times 1}$。对其进行特征值分解，可以得到
$$
R_x(f_n) = U\Lambda U^H = \sum_{i=1}^{M}\lambda_i u_i u_i^H
$$
特征值的大小满足关系 $\lambda_{1} \geqslant \lambda_{2} \geqslant \cdots \geqslant \lambda_{K} > \lambda_{K+1}= \cdots = \lambda_{M}=\sigma^{2}$。 于是，我们可以构造出两个矩阵，$U_S$ 称为信号子空间，$U_N$ 称为噪声子空间。根据 MUSIC 算法，那么平均谱函数为
$$
P(\theta) = \frac{1}{\frac{1}{N}\sum_{n=1}^{N} \Vert\mathbf a^{H}(f_n,\theta) \mathbf{U}_{N}(f_n)\Vert^2}
$$

### 4.2 相干信号子空间算法

因为宽带信号的分解得到的子空间与频率有关，使得不同频点上的子空间会不一样，所以 CSM 算法就是要构造一个聚焦矩阵，通过它**把不同频点的子空间变换到相同的频点**上。同样我们首先需要对阵列接收信号进行 $N$ 点 FFT 分解成 $N$ 个窄带分量，那么聚焦矩阵 $T(f_n)$ 需要满足

$$
T(f_n)A(f_n) = A(f_0),\quad n=1,2,\cdots,N
$$
其中，$f_0$ 为参考频点。于是经过聚焦变换后的阵列接收信号为 
$$
\begin{align*}
Y(f_n) &= T(f_n)X(f_n) \\
&= T(f_n)A(f_n)S(f_n)+T(f_n)N(f_n) \\
&= A(f_0)S(f_n)+T(f_n)N(f_n) \\
\end{align*}
$$
可知，经过变换后的 $Y(f_n)$ **在任何频点上都有着相同的方向矩阵**。于是，我们可以计算出 $Y(f_n)$ 的自相关矩阵： 
$$
\begin{align*}
R_y &= \frac{1}{N}\sum_{n=1}^{N} Y(f_n)Y^H(f_n) \\
&= \frac{1}{N}\sum_{n=1}^{N} T(f_n)X(f_n)X^H(f_n)T^H(f_n) \\
&= \frac{1}{N}\sum_{n=1}^{N}A(f_0)S(f_n)S^H(f_n)A^H(f_0)+
\frac{1}{N}\sum_{n=1}^{N} T(f_n)N(f_n)N^H(f_n)T^H(f_n) \\
&=  A(f_0) \left[ \frac{1}{N} \color{#00F}\sum_{n=1}^{N}S(f_n)S^H(f_n) \right]A^H(f_0)+
\frac{1}{N} \left[\color{#00F}\sum_{n=1}^{N} T(f_n) N(f_n)N^H(f_n) T^H(f_n)\right] \\
&=  A(f_0) \left[\frac{1}{N} \color{#00F}R_S \right]A^H(f_0)+
\frac{1}{N} \left[\color{#00F}R_N \right]
\end{align*}
$$
同样地，我们对构造的矩阵束 $\{ R_y,R_N \}$ 进行广义特征值分解，从而构造出信号子空间 $U_S$ 以及噪声子空间 $U_N$。此时可得谱函数 
$$
P(\theta) = \frac{1}{\Vert\mathbf a^{H}(f_0,\theta) \mathbf{U}_{N}\Vert^2}
$$
接下来给出一种聚焦矩阵的构造方法。首先对所有信源的 DOA 进行预估计，计算出它们的平均值 $\hat{\theta}$，那么聚焦矩阵为 
$$
T = \begin{pmatrix}
a_1(f_0,\hat\theta)/a_1(f_n,\hat\theta) & &\\
& \ddots &\\
& & a_M(f_0,\hat\theta)/a_M(f_n,\hat\theta)
\end{pmatrix}
$$


## 参考文献

[^1]: 《Scatterer Localization Using Large-Scale Antenna Arrays Based on a Spherical Wave-Front Parametric Model》

[^2]:   毫米波低复杂度 DOA 估计与波束成形技术的研究
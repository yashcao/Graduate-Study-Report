# Chan 定位算法
$$
\boxed{\textcolor{blue}{\text{Author:   icaoys@gmail.com}}}
$$
---
Chan 算法是 **TDOA 定位**方法的一个很赞的 trick。但是很多方法一旦从学术的角度去看，就罩上了奇异的光环。TDOA，the time differnces of arrival，**到达时间差**。

Chan 算法[^1]是非递归双曲线方程组解法，具有解析表达式解。其主要的特点为在测量误差服从理想高斯分布时，它的定位精度高、计算量小，并且可以通过增加基站数量来提高算法精度。该算法的推导的前提是基于**测量误差为零均值高斯随机变量**，对于实际环境中误差较大的测量值，比如在有非视距误差的环境下，该算法的性能会有显著下降。Chan 算法在考虑**二维**的情况下，可分为 **只有三个 BS 参与定位** 和 **三个以上 BS 定位** 两种。

## 一、3 个基站

移动台，简称 **tag**，自身发射信号；固定基站，简称 **anchor**，接收来自 tag 的无线信号。

![TDOA](E:\Documents\Markdown\tdoa.png)

**BS1 等基站的坐标已知，为$(x_1,y_1)$、$(x_2,y_2)$、$(x_3,y_3)$。假设第 $i$ 个基站位置为 $(x_i,y_i)$。移动台的位置是未知的，设为 $(x,y)$**。此时设 BS1 为固定站，因此距离差都是以 BS1 为基准。

根据几何关系我们定义以下要用的关系表达式：
$$
r_i = \sqrt{(x_i-x)^2+(y_i-y)^2}   \tag{1}
$$

$$
r_{i,1} =  r_i - r_1 = \sqrt{(x_i-x)^2+(y_i-y)^2} - \sqrt{(x_1-x)^2+(y_1-y)^2} \tag{2}
$$

根据 $(1)$ 另外有如下关系：
$$
\begin{split}
r_i^2 = (x_i-x)^2+(y_i-y)^2 = K_i-2x_ix-2y_iy+x^2+y^2   \\   \boxed{K_i = x_i^2+y_i^2}                  
\end{split}
\tag{3}
$$
根据 $(2)$ 另外有如下关系：
$$
r_i^2 = (r_{i,1}+r_1)^2   \tag{4}
$$
将 $(3)$ 代入 $(4)$，则有:
$$
\begin{split}
r_{i,1}^2+2r_{i,1}r_1+r_1^2=K_i-2x_ix-2y_iy+x^2+y^2   \  \  \Longrightarrow \\
\textcolor{red}{r_{i,1}^2+2r_{i,1}r_1+(x_1-x)^2+(y_1-y)^2=K_i-2x_ix-2y_iy+x^2+y^2}   \  \  \Longrightarrow \\
\textcolor{green}{\boxed{r_{i,1}^2+2r_{i,1}r_1}} = K_i-2x_ix-2y_iy+2x_1x+2y_1y-(x_1^2+y_1^2) = \textcolor{green}{\boxed{r_i^2-r_1^2}}  \  \  \Longrightarrow \\
r_{i,1}^2+2r_{i,1}r_1 = K_i-K_1-2(x_i-x_1)x-2(y_i-y_1)y   \  \  \Longrightarrow \\
\textcolor{blue}{\boxed{r_{i,1}^2+2r_{i,1}r_1 = (K_i-K_1)-2x_{i,1}x-2y_{i,1}y} }\ \ \  ,  \ \ \boxed{x_{i,1} = x_i-x_1}.
\end{split}
\tag{5}
$$


在这里 $(5)$ 是关键一步：**消除了未知数的平方项，仅保留一次项，得到了一系列的线性方程组**。比如当 $i=1,2,3$ 时，有如下表达式：
$$
\begin{equation}  
\left\{
             \begin{array}{**lr**}  
             r_{2,1}^2+2r_{2,1}r_1 = (K_2-K_1)-2x_{2,1}x-2y_{2,1}y &  \\  
             r_{3,1}^2+2r_{3,1}r_1 = (K_3-K_1)-2x_{3,1}x-2y_{3,1}y &    
             \end{array}  
\right.  
\tag{6}
\end{equation}
$$

分析 $(6)$，**首先要明确 $r_{i,1}，K_i，x_{i,1}$ 已知，未知项有 $r_1，x，y$。**首先假设 $r_1$ 是已知的。则 $(6)$ 可以视作**线性方程组**求解。因为是二元一次方程组，因此可以直接利用**消元法**来求解。首先 $(6)$ 可以通过**移项**化简为：
$$
\begin{equation}  
\left\{
             \begin{array}{**lr**}  
             2x_{2,1}x+2y_{2,1}y  = (K_2-K_1)-(r_{2,1}^2+2r_{2,1}r_1) \\
             2x_{3,1}x+2y_{3,1}y =  (K_3-K_1)-(r_{3,1}^2+2r_{3,1}r_1) &    
             \end{array}  
\right.  
\tag{7}
\end{equation}
$$
再简记为：
$$
\begin{equation}  
\left\{
             \begin{array}{**lr**}  
             x_{2,1}x+y_{2,1}y  = C_1 /2 \\
             x_{3,1}x+y_{3,1}y =  C_2 /2&    
             \end{array}  
\right.  
\tag{8}
\end{equation}
$$
在这里可以视为 $\left[ \begin{matrix} x_{2,1} & y_{2,1}  \\ x_{3,1} & y_{3,1} \end{matrix} \right] \left[ \begin{matrix} x   \\  y \end{matrix} \right] = \left[ \begin{matrix} \frac{C_1}{2}   \\  \frac{C_2}{2} \end{matrix} \right] $，即 $Ax=C$。可以利用**矩阵的逆**即 $x = A^{-1}C$，也可以利用如下的行列式除法。最后求得：
$$
\begin{equation}  
\left\{
             \begin{array}{**lr**}
             x = \frac{y_{3,1}C_1-y_{2,1}C_2}{2(x_{2,1}y_{3,1}-x_{3,1}y_{2,1})} \\
             y = \frac{x_{2,1}C_2-x_{3,1}C_1}{2(x_{2,1}y_{3,1}-x_{3,1}y_{2,1})} &    
             \end{array}  
\right.  
\text{review:}\quad  x = \frac{\vert{D_1}\vert}{\vert{D}\vert},  \ \ \vert{D1}\vert =   
\left|\begin{array}{cccc}   
    b_1 &    a_{12}     \\   
    b_2 &    a_{22}   
     
\end{array}\right|   

\tag{9}
\end{equation}
$$
其中，$C_1= (K_2-K_1)-(r_{2,1}^2+2r_{2,1}r_1)$。将 $C_1，C_2$ 代入 $(9)$ 可得：
$$
\begin{equation}  
\left\{
             \begin{array}{**lr**}
             x = \frac{y_{2,1}r_{3,1}^2-y_{3,1}r_{2,1}^2+y_{3,1}(K_2-K_1)-y_{2,1}(K_3-K_1)+2(y_{2,1}r_{3,1}-y_{3,1}r_{2,1})r_1}{2(x_{2,1}y_{3,1}-x_{3,1}y_{2,1})} \\
             y =  \frac{x_{3,1}r_{2,1}^2-x_{2,1}r_{3,1}^2+x_{2,1}(K_3-K_1)-x_{3,1}(K_2-K_1)+2(x_{3,1}r_{2,1}-x_{2,1}r_{3,1})r_1}{2(x_{2,1}y_{3,1}-x_{3,1}y_{2,1})} &    
             \end{array}  
\right.  
\tag{10}
\end{equation}
$$
对于 $(10)$ 进一步处理得：
$$
\begin{equation}  
\left\{
             \begin{array}{**lr**}
             x = \frac{y_{2,1}r_{3,1}^2-y_{3,1}r_{2,1}^2+y_{3,1}(K_2-K_1)-y_{2,1}(K_3-K_1)+2(y_{2,1}r_{3,1}-y_{3,1}r_{2,1})r_1}{2(x_{2,1}y_{3,1}-x_{3,1}y_{2,1})} \\
             y =  \frac{x_{2,1}r_{3,1}^2-x_{3,1}r_{2,1}^2+x_{3,1}(K_2-K_1)-x_{2,1}(K_3-K_1)+2(x_{2,1}r_{3,1}-x_{3,1}r_{2,1})r_1}{2(x_{3,1}y_{2,1}-x_{2,1}y_{3,1})} &    
             \end{array}  
\right.  
\tag{11}
\end{equation}
$$
其中，我们令：
$$
\begin{split}
p_1 = \frac{y_{2,1}r_{3,1}^2-y_{3,1}r_{2,1}^2+y_{3,1}(K_2-K_1)-y_{2,1}(K_3-K_1)}{2(x_{2,1}y_{3,1}-x_{3,1}y_{2,1})} \\
q_1 \cdot r_1 = \frac{2(y_{2,1}r_{3,1}-y_{3,1}r_{2,1})r_1}{2(x_{2,1}y_{3,1}-x_{3,1}y_{2,1})}  \Longrightarrow q_1  = \frac{y_{2,1}r_{3,1}-y_{3,1}r_{2,1}}{x_{2,1}y_{3,1}-x_{3,1}y_{2,1}} \\
p_2 =  \frac{x_{2,1}r_{3,1}^2-x_{3,1}r_{2,1}^2+x_{3,1}(K_2-K_1)-x_{2,1}(K_3-K_1)}{2(x_{3,1}y_{2,1}-x_{2,1}y_{3,1})} \\
q_2 \cdot r_1 = \frac{2(x_{2,1}r_{3,1}-x_{3,1}r_{2,1})r_1}{2(x_{3,1}y_{2,1}-x_{2,1}y_{3,1})}  \Longrightarrow q_2  = \frac{x_{2,1}r_{3,1}-x_{3,1}r_{2,1}}{x_{3,1}y_{2,1}-x_{2,1}y_{3,1}}
\end{split}
\tag{12}
$$

利用 $(12)$ 我们就可以将 $(11)$ 表示为如下非常简洁的形式：
$$
\begin{equation}  
\left\{
             \begin{array}{**lr**}
             x = p_1+q_1r_1 \\
             y = p_2+q_2r_1   &    
             \end{array}  
\right.  
\tag{13}
\end{equation}
$$
其中，**只有 $r_1$ 是未知的**。因此要想求出 MS 的坐标 $(x,y)$，由式 $(13)$ 可知，首先要求出 $r_1$。接下来我们将式$(13)$ 代入到 式 $(3)  \ \ \textcolor{blue}{\boxed{r_1^2 = K_1-2x_1x-2y_1y+x^2+y^2 \quad  (\text{let :} \ \  i=1)} }$，可以化为如下形式：

$$
\begin{split}
(q_1^2+q_2^2-1) \cdot r_1^2 -2[q_1(x_1-p_1)+q_2(y_1-p_2)] \cdot r_1+[(x_1-p1)^2+(y_1-p_2)^2]=0 \\
\Longrightarrow \boxed{a \cdot r_1^2+b \cdot r_1+c=0}
\end{split}
\tag{14}
$$
式 $(14)$ 是以 $r_1$ 的一元二次方程。求解式 $(14)$ 可得 $r_1$ 的两个根，根据**先验信息可以舍去一个无效的根**，将有效的 $r_1$ 代回到式 (13) 可求可求出 MS 的估计坐标。

```matlab
% 求解一元二次方程
syms r;
r0 = solve(a*r^2+b*r+c); % 默认 eqx=0
% select the ture 'r0', then solve the x & y.
```

## 二、多个基站

Chan 算法是基于双曲线交点的定位方法，闭合解/解析解，小范围和大范围的定位系统都适用。当 TDOA 估计误差较小时，可以认为是 ML（最大似然法）的一种近似方法。

当基站的数量大于 3 时，TDOA 值得到的**非线性方程组个数要多于未知变量的个数**。采用加权最小二乘法(WLS)来充分利用冗余的数据，Chan 算法能获得更加好的 MS 位置估计值。此时**先将初始非线性 TDOA 方程组转换为线性方程组，然后采用 WLS 得到初始解，再利用第一次得到的估计坐标及附加变量等已知约束条件进行第二次 WLS 估计，从而得到改进的估计坐标**。
### 2.1 问题建模
当有更多的 anchor 时，线性方程组便是超定的。由于测量噪声的存在 $u(k) = s(d-k)+n(k)$，方程组 $(5)$ 的解不可能完全满足其中的所有方程，所以合适的解只能是匹配这些方程组的最佳解。令 $\mathbf{z} = \left[ \begin{matrix} x & y & r_1  \end{matrix} \right]^ \mathrm{T}$，在这里先不考虑 $x ， y ， r_1$ 三者之间的关系，即假设三者线性无关。建立存在 TDOA 噪声的**线性方程组**，将式 $(5)$ 转变成矩阵形式如下：
$$
\begin{split}
r_i^2-r_1^2 = r_{i,1}^2+2r_{i,1}r_1 = (K_i-K_1)-2x_{i,1}x-2y_{i,1}y   \  \  \Longrightarrow \\
r_{i,1}^2+2r_{i,1}r_1 = -2x_{i,1}x-2y_{i,1}y+K_i-K_1    \Longrightarrow  \textcolor{blue}{\boxed{r_{i,1}^2-K_i+K_1 = -2x_{i,1}x-2y_{i,1}y-2r_{i,1}r_1}}   \Longrightarrow  \\
\frac{1}{2}(r_{i,1}^2-K_i+K_1)  = \left[ \begin{matrix} -x_{i,1} & -y_{i,1} & -r_{i,1} \end{matrix} \right] \left[ \begin{matrix} x \\ y \\r_1  \end{matrix} \right]   \Longrightarrow     \\  
\mathbf{h} =  \frac{1}{2} \left[ \begin{matrix} r_{2,1}^2-K_2+K_1 \\ r_{3,1}^2-K_3+K_1 \\…  \end{matrix} \right]   \text{and}  \ \ \mathbf{G} = -{\left[ \begin{matrix} x_{2,1} & y_{2,1}&r_{2,1}  \\ x_{3,1} & y_{3,1}&r_{3,1} \\…& …&…  \end{matrix} \right]}     \Longrightarrow \ \  \mathbf{h} = \mathbf{G z^0}\  \textcolor{red}{ (\mathbf{z^0}\  \text{is the groundturth.})}\\    
\end{split}
\tag{15}
$$


在这里，考虑存在 TDOA 观测噪声时即（$\mathbf{G z^0}\neq \mathbf{h}$），误差向量为
$$
\mathbf{h} = \mathbf{G z^0}+\psi  \Longrightarrow \textcolor{red}{\boxed{ \psi = \mathbf{h} - \mathbf{G z^0}}}   \tag{16}
$$
### 2.2 关于最小二乘法
为保证参数估计量具有良好的性质，通常对模型提出若干基本假设。实际这些假设与所采用的估计方法紧密相关。估计方法有多种，其中使用最广泛的是普通最小二乘法（ordinary least squares, OLS）。 

>线性回归模型为 $Y=X\beta+e$，模型的**基本假设**有：

> -   假设1、解释变量 $X$ 是确定性变量，不是随机变量；
> -   **假设2**、随机误差项 $e$ 具有**零均值**、**同方差**和**不相关性**：
$$
\begin{split}
\mathbb{E}(e_i)=0    \qquad \qquad \qquad \qquad \qquad         i=1,2, …,n   \\
\text{Var} (e_i)=\sigma^2     \qquad \qquad \qquad \quad \  \  \quad          i=1,2, …,n   \\
\text{Cov} (e_i, e_j)=0   \ \ \ \ \ \ \ \        i≠j  \ \ \text{and} \ \  i,j= 1,2, …,n
\end{split}
$$
> -   假设3、随机误差项 $e$ 与解释变量 $X$ 之间不相关：
$$
\text{Cov}(X_i, e_i)=0      \qquad \qquad \qquad \  \ \   \ \  i=1,2, …,n
$$
> -   **假设4**、$e$ 服从**零均值**、**同方差**、**零协方差**的**正态分布**：
$$
e_i \sim  N(0, \sigma_e^2 )    \qquad \qquad \qquad   \  \ \   \ \  \ \   \   i=1,2, …,n
$$

需要注意的是，如果假设 1、2 满足，则假设 3 也满足；**如果假设 4 满足，则假设 2 也满足**。估计参数时，必须检验基本假定是否满足，并针对基本假定不满足的情况，采取相应的补救措施或者新的方法。不满足基本假设，**高斯-马尔科夫定理**[^5]失效。 高斯-马尔可夫定理表示为：

$$
\textcolor{blue}{\boxed{\mathbb{E}(\mathbf{e})=0 \qquad \text{Cov}(\mathbf{e})=\sigma ^{2}\mathbf{I_n} \qquad \text{Cov}(\mathbf{e,X})=0}}
$$

当出现**异方差**，即 $\text{Var} (e_i)=\sigma_i^2,     \quad   i=1,2, …,n$，不满足上述定理时，OLS 估计量虽具有无偏性，但不具有有效性[^6]。因此采用加权最小二乘法，对原 OLS 模型进行改进，使之成为一个新的不存在异方差的模型，再利用 OLS 解决。

### 2.3 加权最小二乘法
我们需要先了解一下加权最小二乘法(WLS)。所谓最小二乘法，广泛用于数据拟合的求解方法，优化思路是使得估计误差（residual, $R$）的平方和最小化。加权最小二乘法考虑到某些观测值具有更高的权重（如误差小），则问题转化为 ${\lVert R\rVert }_{w}^{2}=\sum_{i=1}^{M}w_i^2r_i^2$。其中 $w$ 是**权重**。

最普通的最小二乘法的回归模型，满足上述的高斯-马尔可夫定理时，即可通过对 $\beta$ 求导[^7]，得到其估计值：
$$
\begin{split}
Y=X\beta+e \\
R(\beta)={\lVert Y-X\beta\rVert }^{2}=(Y-X\beta)^{\mathrm{T}}(Y-X\beta) \\
\boxed{\hat{\beta} = (X^{\mathrm{T}}X)^{-1}X^{\mathrm{T}}Y}
\end{split}
\tag{17}
$$
当残差项 $e$ 的方差[^2] $\text{var}(e)$ 不再是已知的，且不再是互不相关的，即 $\mathbb{E}(e)=0，\text{Var}(e)={\sigma}^2{\Sigma} $。用 WLS 改进原模型，使之不存在异方差即 $\textcolor{purple}{\sum_i W_i(Y_i-X_i \beta)^2=\sum_i W_ie_i^2}$，因此有：
$$
\begin{split}
\sum_i W_i(Y_i-X_i \beta)^2=\sum_i W_ie_i^2=\mathbf{e}^{\mathrm{T}}W\mathbf{e}     \\
R(\beta)=(Y-X\beta)^{\mathrm{T}}{\Sigma}^{-1}(Y-X\beta) \\
\textcolor{red}{\boxed{\hat{\beta} = (X^{\mathrm{T}}{\Sigma}^{-1}X)^{-1}X^{\mathrm{T}}{\Sigma}^{-1}Y}}
\end{split}
\tag{18}
$$
加权最小二乘法，就是对上述的 ${\Sigma}$ 取一种特殊的矩阵：**对角阵**，而这个对角阵的**对角元都是常数**，也就是**权重的倒数**，如下： 
$$
\begin{split}
\text{Cov}(e)=\text{Var}(e)={\sigma}^2{\Sigma}={\sigma}^2\begin{bmatrix} 1/w_1 &…& 0& 0 \\ 0 & 1/w_2&…& 0\\ …&…&…&…\\0&…& 0&1/w_n \end{bmatrix}
\end{split}
\tag{19}
$$
权重的选取原则是：对于较大的残差平方赋予较小的权重，反之赋予较大的权重。令 $W={\Sigma}^{-1}$ 表示权重矩阵。令 $W$ 的**每个元素都开平方根**作为新的矩阵 $F_w$，有 $F_wY=F_wX\beta+F_w\mathbf{e} \Longrightarrow\text{Var}(F_w\mathbf{e})={\sigma}^2W{\Sigma}={\sigma}^2\mathbf{I_n}$ 。在这里特别注意，回归模型已经改变[^8]，则新的模型以及式 $(18)$ 中的残差变为：
$$
\begin{split}
\textcolor{red}{\boxed{F_wY=F_wX\beta+F_w\mathbf{e} \Longrightarrow Y^*=X^*\beta+\mathbf{e}^* }}  \\
R(\beta)=(Y-X\beta)^{\mathrm{T}}W(Y-X\beta) \\
\boxed{\hat{\beta} = (X^{\mathrm{T}}WX)^{-1}X^{\mathrm{T}}WY}
\end{split}
\tag{20}
$$

其中，$W$ 是未知的。首先要解决未知的 $\Sigma$。在论文[^1]中，${\psi = \mathbf{h} - \mathbf{G z}=C\mathbf{Bn}+0.5C^2\mathbf{n}\odot \mathbf{n}}$，符号 $\odot$ 是 Hadamard 乘积 [^3]。$\psi$ 的推导原理见附录[^9]。

再看式 $(15)$ 中的 $r_{i,1}=r_{i,1}^0+Cn_{i,1}, \ \  r_{i,1}^0 = r_{i}^0-r_{1}^0$，注意 $n_{i,1}$ 是**时延误差**，$\mathbf{n}$ 的均值为 $\mathbf{0}$，协方差矩阵是 $\mathbf{Q}$。$C$ 为信号的传播速度。实际情况下又满足 $Cn_{i,1}<<r_i^0$，因此可以**忽略**式 $\textcolor{green}{\boxed{\psi =C\mathbf{Bn}+0.5C^2\mathbf{n}\odot \mathbf{n}}}$ 中的**第二项**，则误差 $\psi$ 变为**高斯随机向量**。由于 $(16)$ 服从零均值高斯随机分布，我们将 $(16)$ 和 $(20)$ 联系起来，则可以得到：
$$
\textcolor{blue}{\boxed{\hat{\mathbf{z}} = (\mathbf{G}^{\mathrm{T}}W\mathbf{G})^{-1}\mathbf{G}^{\mathrm{T}}W\mathbf{h}}}
\tag{21}
$$

在这里假设共有 $M$ 个 BS，误差 $\psi$ 的协方差矩阵[^4]为： 
$$
\Sigma = \text{Cov}(\psi,\psi)=\mathbb{E}(\psi\psi^{\mathrm{T}})=C^2\mathbf{BQB}
\tag{22}
$$
其中 $\mathbf{B}=\text{diag}(r_2^0, r_3^0 ,…,r_M^0)$，因此 $\mathbf{B}^{\mathrm{T}}=\mathbf{B}$，$\mathbf{Q}=\mathbb{E}(\mathbf{nn}^{\mathrm{T}})=\text{Cov}(\mathbf{n})=\mathbf{I}_{M-1}$[^10]。因为 $\mathbf{B}$ 中有 MS 到 BS 的距离，在计算时是**未知**的。**如果基站与源很远很远**，可以合理地认为 $r_i^0$ 都接近于同一个常数即假设 $\mathbf{B}\approx r_a\mathbf{I}$，则 $\Sigma \approx C^2r_a^2\mathbf{Q}$。论文[^1]提示由于 $(22)$ 的 $\Sigma$ 的**缩放不影响其结果**[^8]，我们可以得到 $(21)$ 的近似：

$$
\textcolor{red}{\boxed{\widetilde{\mathbf{z}} \approx (\mathbf{G}^{\mathrm{T}}\mathbf{Q}^{-1}\mathbf{G})^{-1}\mathbf{G}^{\mathrm{T}}\mathbf{Q}^{-1}\mathbf{h}}}
\tag{23}
$$
**如果基站与源较近**，可以利用 $(23)$ 求得的 $\widetilde{\mathbf{z}}$ 代入式 $(1)$ 重新估计 $\mathbf{B}$，继而得到 $\Sigma$，再代入式 $(21)$ 便可以得到在忽略元素 $\mathbf{z}$ 间约束关系下的估计结果。

这里求得的 $\hat{\mathbf{z}}$ 是在假定 $\mathbf{z}$ 的各元素独立的情况下进行的，而实际上 $\mathbf{z}$ 中 $r_1$ 是与 $(x,y)$ 有关的量，用 $\mathbf{Q}$ 近似地代替 $\Sigma$ 误差矢量的协方差矩阵会带来一定的误差。为了得到更精确的定位结果，可以接着进行第二次 WLS 估计。

注意到 $r_1$ 与 $(x,y)$ 始终约束到式 $(3)  \ \ \textcolor{blue}{\boxed{r_1^2 = K_1-2x_1x-2y_1y+x^2+y^2 } }$，那么，能否利用这个关系，更进一步的提高估计精度？ 假设在有噪声的情况下，当 TDOA 的测量噪声足够小的时候，向量 $\mathbf{z}$ 是一个随机向量，其均值为 groundtruth，它的元素表示如下：
$$
\begin{split}
\mathbf{z}_1=x^0+e_1 \\
\mathbf{z}_2=y^0+e_2 \\
\mathbf{z}_3=r_1^0+e_3
 \end{split} \tag{24}
$$
这里特别的处理是从向量 $\bold{z}$ 的前两个元素中分别减去 $x_1$ 和 $y_1$（anchor 1 的坐标），再对向量 $\bold{z}$ 的元素进行平方运算，见如下公式： 

$$
\begin{split}
\mathbf{h'}=\left[ \begin{matrix} (z_1-x_1)^2 \\ (z_2-y_1)^2 \\z_3^2  \end{matrix} \right] \  \text{and}  \ \ \mathbf{G'} = {\left[ \begin{matrix} 1 & 0  \\ 0 & 1 \\1 & 1   \end{matrix} \right]}\ \  \text{and}  \ \ \mathbf{z'} = {\left[ \begin{matrix} (x-x_1)^2  \\ (y-y_1)^2   \end{matrix} \right]}  \\      \Longrightarrow   
\textcolor{red}{\boxed{ \psi ' = \mathbf{h'}-\mathbf{G' {z'}^0} }}\\
\mathbf{G' {z'}^0}={\left[ \begin{matrix} 1 & 0  \\ 0 & 1 \\1 & 1   \end{matrix} \right]} {\left[ \begin{matrix} (x^0-x_1)^2  \\ (y^0-y_1)^2   \end{matrix} \right]}=\left[ \begin{matrix} (x^0-x_1)^2 \\ (y^0-y_1)^2 \\{(r_1^0)}^2  \end{matrix} \right] \\
 \Longrightarrow  \mathbf{h'}-\mathbf{G' {z'}^0}=\left[ \begin{matrix} (x^0+e_1-x_1)^2 \\ (y^0+e_2-y_1)^2 \\{(r_1^0+e_3)}^2  \end{matrix} \right]-\left[ \begin{matrix} (x^0-x_1)^2 \\ (y^0-y_1)^2 \\{(r_1^0)}^2  \end{matrix} \right]
\end{split} \tag{25}
$$

从式 $(25)$ 可以看出，新的误差矢量 $\psi ' $ 表明了向量 $\mathbf{z}$ 的**不准确性**。将式 $(24)$ 代入式 $(25)$，则有：
$$
\begin{split}
{\psi_1}' =2(x^0-x_1)e_1+e_1^2 \approx 2(x^0-x_1)e_1 \\
{\psi_2}' =2(y^0-y_1)e_2+e_2^2 \approx 2(y^0-y_1)e_2 \\
{\psi_3}' =2r_1^0 e_3+e_3^2 \approx 2r_1^0 e_3
 \end{split} \tag{26}
$$
当式 $(26)$ 中的 $e_i$ 较小时，近似成立。现在忽略掉 $w_1^2$ 平方项，只考虑近似后的 $\psi ' $ 矢量，和前面的处理过程类似，比较容易推出：
$$
\begin{split}
\Sigma' = \mathbb{E}({\psi}' {\psi}'^{\mathrm{T}})  =\mathbb{E}\left ( \left[ \begin{matrix} 2(x^0-x_1)e_1 \\ 2(y^0-y_1)e_2 \\2r_1^0 e_3  \end{matrix} \right]\left[ \begin{matrix} 2(x^0-x_1)e_1 \\ 2(y^0-y_1)e_2 \\2r_1^0 e_3  \end{matrix} \right]^\mathrm{T}  \right )  \\
=4\mathbb{E} \left (\left[ \begin{matrix} (x^0-x_1)^2e_1^2&(x^0-x_1)(y^0-y_1)e_1e_2&(x^0-x_1)r_1^0e_1e_3 \\ (x^0-x_1)(y^0-y_1)e_1e_2 &(y^0-y_1)^2e_2^2&(y^0-y_1)r_1^0e_1e_3\\(x^0-x_1)r_1^0e_1e_3&(y^0-y_1)r_1^0e_1e_3&{r_1^0}^2e_3^2  \end{matrix} \right]   \right )                                      \\
=4\left[ \begin{matrix} x^0-x_1&&\\&y^0-y_1&\\&&{r_1^0}  \end{matrix} \right] \mathbb{E}\left (\left[ \begin{matrix} e_1\\e_2\\e_3  \end{matrix} \right] \left[ \begin{matrix} e_1&e_2&e_3  \end{matrix} \right]  \right ) \left[ \begin{matrix} x^0-x_1&&\\&y^0-y_1&\\&&{r_1^0}  \end{matrix} \right]  \\     \Longrightarrow
\textcolor{magenta}{\boxed{\Sigma'=4 \cdot \mathbf{B'}\cdot \text{Cov}\mathbf{(z)}\cdot \mathbf{B'}}}
\end{split}
\tag{27}
$$

其中，$\mathbf{B'}=\text{diag}(x^0-x_1, y^0-y_1, {r_1^0})$。这样得到了 $\psi ' $ 的近似的协方差矩阵 $\Sigma'$。误差 $\psi ' $ 为**高斯随机向量**，因此同样可以采用 $(20)$ 的方法进行估计：
$$
\textcolor{blue}{\boxed{\hat{\mathbf{z'}} = (\mathbf{G'}^{\mathrm{T}}{\Sigma'}^{-1}\mathbf{G'})^{-1}\mathbf{G'}^{\mathrm{T}}{\Sigma'}^{-1}\mathbf{h'}}}
\tag{28}
$$

注意到式 $(27)$ 的 $\Sigma'$ 未知，首先计算估计 $\mathbf{z}$ 的协方差矩阵，采用**一阶扰动方法计算和保留线性扰动分量**，找到式 $(27)$ 中的 $\text{Cov}\mathbf{(z)}$。注意式 $(15)$ 中的 $\mathbf{G、h}$ 都包含随机变量 $r_{i,1}$，则 $\mathbf{G}=\mathbf{G}^0+\Delta\mathbf{G},  \quad \mathbf{h}=\mathbf{h}^0+\Delta\mathbf{h}$。由于式 $(15)$ 满足 $\mathbf{G ^0z^0}=\mathbf{h}^0$，则式 $(16)$ 的 $\psi$ 可以重写为：
$$
\psi =\Delta\mathbf{h}-\Delta\mathbf{Gz^0} \tag{29}
$$
式 $(24)$ 记为 $\mathbf{z}=\mathbf{z^0}+\Delta\mathbf{z}$，根据公式 $(21)$ 有：
$$
\begin{split}
\mathbf{z}=(\mathbf{G}^{\mathrm{T}}W\mathbf{G})^{-1}\mathbf{G}^{\mathrm{T}}W\mathbf{h} \qquad  \Longrightarrow \\
\mathbf{z^0}+\Delta\mathbf{z}=  [(\mathbf{G}^0+\Delta\mathbf{G})^{\mathrm{T}}W (\mathbf{G}^0+\Delta\mathbf{G})]^{-1}(\mathbf{G}^0+\Delta\mathbf{G})^{\mathrm{T}}W(\mathbf{h}^0+\Delta\mathbf{h})    \\
\textcolor{purple}{
 [(\mathbf{G}^0+\Delta\mathbf{G})^{\mathrm{T}}W (\mathbf{G}^0+\Delta\mathbf{G})] (\mathbf{z^0}+\Delta\mathbf{z})  =  (\mathbf{G}^0+\Delta\mathbf{G})^{\mathrm{T}}W(\mathbf{h}^0+\Delta\mathbf{h})  }   \\ 
\mathbf{G ^0z^0}+(\mathbf{G ^0}\Delta\mathbf{z}+\Delta\mathbf{G}\mathbf{z^0}+\Delta\mathbf{G}\Delta\mathbf{z})
=\mathbf{h^0}+\Delta\mathbf{h}
\end{split}
\tag{30}
$$

保留一阶扰动分量，并略去高阶扰动量。结合公式 $\textcolor{green}{\boxed{\psi =C\mathbf{Bn}}}​$ 和公式 $(29)​$ ， 利用式 $(22)​$ 得到 $\Delta\mathbf{z}​$ 和 $\mathbf{z}​$ 的协方差矩阵的表达式：

$$
\begin{split}
\Delta\mathbf{h}=\mathbf{G ^0}\Delta\mathbf{z}+\Delta\mathbf{G}\mathbf{z^0}+\Delta\mathbf{G}\Delta\mathbf{z} \approx \mathbf{G ^0}\Delta\mathbf{z}+\Delta\mathbf{G}\mathbf{z^0} \\
\psi =\Delta\mathbf{h}-\Delta\mathbf{Gz^0}=\mathbf{G ^0}\Delta\mathbf{z}+\Delta\mathbf{G}\mathbf{z^0}-\Delta\mathbf{Gz^0}=\mathbf{G ^0}\Delta\mathbf{z} \\
\psi =\mathbf{G ^0}\Delta\mathbf{z}=C\mathbf{Bn} \quad \Longrightarrow \quad \textcolor{blue}{\boxed{e =C\mathbf{Bn}-\mathbf{G ^0}\Delta\mathbf{z}}} \\
\overset{\textcolor{red}{\text{WLS}}}{\Longrightarrow} \text{arg min}[(C\mathbf{Bn}-\mathbf{G ^0}\Delta\mathbf{z})^{\mathrm{T}}W(C\mathbf{Bn}-\mathbf{G ^0}\Delta\mathbf{z})] \\
\Longrightarrow \Delta\mathbf{z}=(\mathbf{G^0}^{\mathrm{T}}W\mathbf{G^0})^{-1}\mathbf{G^0}^{\mathrm{T}}WC\mathbf{Bn} \\
\Longrightarrow \textcolor{blue}{\boxed{\Delta\mathbf{z}=C(\mathbf{G^0}^{\mathrm{T}}W\mathbf{G^0})^{-1}\mathbf{G^0}^{\mathrm{T}}W\mathbf{Bn}  }}  \\
\end{split}
\tag{31}
$$

$$
\begin{align*} 
\text{Cov}\mathbf{(z)} & = \mathbb{E}(\Delta\mathbf{z}\Delta\mathbf{z}^{\mathrm{T}})  \\
& =\mathbb{E}[C^2(\mathbf{G^0}^{\mathrm{T}}W\mathbf{G^0})^{-1}\mathbf{G^0}^{\mathrm{T}}W\mathbf{Bn}\cdot \mathbf{n}^{\mathrm{T}}\mathbf{B}^{\mathrm{T}}W^{\mathrm{T}}\mathbf{G^0} \{ (\mathbf{G^0}^{\mathrm{T}}W\mathbf{G^0})^{-1}\}^{\mathrm{T}} ] \\
& \underrightarrow{\textcolor{red}{\mathbf{B}^{\mathrm{T}}=\mathbf{B},\ W^\mathrm{T}=W, \ \Sigma =W^{-1}=C^2\mathbf{BQB} } }    \\
&= C^2(\mathbf{G^0}^{\mathrm{T}}W\mathbf{G^0})^{-1}\mathbf{G^0}^{\mathrm{T}}W \mathbf{B} \cdot \textcolor{purple}{\mathbb{E} [\mathbf{n} \mathbf{n}^{\mathrm{T}}] }\cdot \mathbf{B}^{\mathrm{T}}W^{\mathrm{T}}\mathbf{G^0} \{ (\mathbf{G^0}^{\mathrm{T}}W\mathbf{G^0})^{-1}\}^{\mathrm{T}}  \\
&=(\mathbf{G^0}^{\mathrm{T}}W\mathbf{G^0})^{-1}\mathbf{G^0}^{\mathrm{T}}W \textcolor{purple}{ C^2 \mathbf{B}  \mathbf{Q} \mathbf{B} } W^{\mathrm{T}}\mathbf{G^0} \{ (\mathbf{G^0}^{\mathrm{T}}W\mathbf{G^0})^{-1}\}^{\mathrm{T}}  \\
&=(\mathbf{G^0}^{\mathrm{T}}W\mathbf{G^0})^{-1}\mathbf{G^0}^{\mathrm{T}}W \textcolor{purple}{W^{-1} }W^{\mathrm{T}}\mathbf{G^0} \{ (\mathbf{G^0}^{\mathrm{T}}W\mathbf{G^0})^{-1}\}^{\mathrm{T}} \\
&=(\mathbf{G^0}^{\mathrm{T}}W\mathbf{G^0})^{-1}\mathbf{G^0}^{\mathrm{T}}W \mathbf{G^0} \{ (\mathbf{G^0}^{\mathrm{T}}W\mathbf{G^0})^{-1}\}^{\mathrm{T}} \\
&=\textcolor{green}{ (\mathbf{G^0}^{\mathrm{T}}W\mathbf{G^0})^{-1}  (\mathbf{G^0}^{\mathrm{T}}W \mathbf{G^0})} \{ (\mathbf{G^0}^{\mathrm{T}}W\mathbf{G^0})^{-1}\}^{\mathrm{T}} \\
&=\{ (\mathbf{G^0}^{\mathrm{T}}W\mathbf{G^0})^{-1}\}^{\mathrm{T}} \\
& \underrightarrow{\textcolor{red}{[A^{-1}]^T = [A^{T}]^{-1}  } } \\
&=\{ (\mathbf{G^0}^{\mathrm{T}}W\mathbf{G^0})^{\mathrm{T}}\}^{-1} \\
&=\textcolor{blue}{\boxed{(\mathbf{G^0}^{\mathrm{T}}W\mathbf{G^0})^{-1}}}
\end{align*}
\tag{32}
$$

需要明确一个问题，在式 $(32)$ 的推导中，期望和（协）方差矩阵一旦计算出来便可以视作常量矩阵，因此可以提出来作为系数因子。式 $(32)$ 中的 $\mathbf{G^0}$、$W$ 、$\mathbf{B}$ 都是常量矩阵。

回到式 $(27)$，协方差矩阵 $\Sigma'$ 中包含了 $\mathbf{B'}=\text{diag}(x^0-x_1, y^0-y_1, {r_1^0})$，我们发现里面包含了**未知**的 groundtruth。因此不能利用式 $(28)$ 直接计算，同理利用上一次 WLS 的式 $(21)$ 的结果 $\hat{\mathbf{z}}$ 去近似计算 $\mathbf{B'}$。另外通过式 $(32)$ 得到了 $\text{Cov}\mathbf{(z)}$ 的表达式，其中未知的 $\mathbf{G^0}$ 我们可以用式 $(15)$ 的 $\mathbf{G}$ 近似。

**考虑源距离很远的条件下**，即 $\mathbf{B}\approx r_a\mathbf{I}$， $\Sigma \approx C^2r_a^2\mathbf{Q}$。式 $(32)$ 近似为 $\text{Cov}\mathbf{(z)} =C^2r_a^2(\mathbf{G^0}^{\mathrm{T}}\mathbf{Q}^{-1}\mathbf{G^0})^{-1}$。于是式 $(28)$ 近似为：
$$
\begin{align*}
\Sigma' &\approx  4 \cdot \mathbf{B'}\cdot C^2r_a^2(\mathbf{G^0}^{\mathrm{T}}\mathbf{Q}^{-1}\mathbf{G^0})^{-1}  \cdot \mathbf{B'} \\
&=  4C^2r_a^2\mathbf{B'}(\mathbf{G^0}^{\mathrm{T}}\mathbf{Q}^{-1}\mathbf{G^0})^{-1}\mathbf{B'} \\
&\approx  4C^2r_a^2\mathbf{B'}(\mathbf{G}^{\mathrm{T}}\mathbf{Q}^{-1}\mathbf{G})^{-1}\mathbf{B'}  \\
\textcolor{red}{\text{let}} \quad \Sigma' &=  \mathbf{B'}(\mathbf{G}^{\mathrm{T}}\mathbf{Q}^{-1}\mathbf{G})^{-1}\mathbf{B'}    \\
\Sigma' ^{-1}& = \mathbf{B'}^{-1} \mathbf{G}^{\mathrm{T}}\mathbf{Q}^{-1}\mathbf{G} \mathbf{B'}^{-1} \\
\hat{\mathbf{z'}}&= (\mathbf{G'}^{\mathrm{T}}{\Sigma'}^{-1}\mathbf{G'})^{-1}\mathbf{G'}^{\mathrm{T}}{\Sigma'}^{-1}\mathbf{h'} \\
&\approx   (\mathbf{G'}^{\mathrm{T}}  \mathbf{B'}^{-1} \mathbf{G}^{\mathrm{T}}\mathbf{Q}^{-1}\mathbf{G} \mathbf{B'}^{-1} \mathbf{G'})^{-1}\mathbf{G'}^{\mathrm{T}}  \mathbf{B'}^{-1} \mathbf{G}^{\mathrm{T}}\mathbf{Q}^{-1}\mathbf{G} \mathbf{B'}^{-1} \mathbf{h'}
\end{align*}
\tag{33}
$$
注意矩阵 $\mathbf{G'}$ 是常量。在这里，我们顺便考虑一下 $\mathbf{z'}$ 的协方差矩阵。通过计算 $\mathbf{z'}$ 以及 $\mathbf{z'}\mathbf{z'}^{T}$ 的期望，结合公式 $(25)$ 和公式 $(32)$ 进一步可得：
$$
\begin{align*} 
 \psi = \mathbf{h} - \mathbf{G z^0} \Longrightarrow \psi &=\mathbf{G^0}\Delta  \mathbf{z}   \\
\text{Cov}\mathbf{(z)}  &= \textcolor{green}{\boxed{(\mathbf{G^0}^{\mathrm{T}}{\Sigma}^{-1}\mathbf{G^0})^{-1}}} \\
 \underrightarrow{\textcolor{red}{\text{likewise}  } } \\
\psi ' = \mathbf{h'}-\mathbf{G' {z'}^0}  \Longrightarrow    \Delta \mathbf{{h'}} &= \textcolor{green}{\boxed{\mathbf{G'}\Delta  \mathbf{z'} =\psi ' }} \\
\text{Cov}\mathbf{({\psi}')}=\Sigma' \Longrightarrow \Delta  \mathbf{z'} &=(\mathbf{{G'}}^{\mathrm{T}}{\Sigma'}^{-1}\mathbf{G'})^{-1}\mathbf{{G'}}^{\mathrm{T}}{\Sigma'}^{-1}{\psi}' \\
\text{Cov}\mathbf{(z')}  &= \mathbb{E}(\Delta  \mathbf{z'}\Delta  \mathbf{{z'}}^{\mathrm{T}})  \\
&= (\mathbf{{G'}}^{\mathrm{T}}{\Sigma'}^{-1}\mathbf{G'})^{-1}\mathbf{{G'}}^{\mathrm{T}}{\Sigma'}^{-1}\mathbb{E} ( {\psi}' {\psi}'^{\mathrm{T}}) {\Sigma'}^{-1} \mathbf{{G'}}  (\mathbf{{G'}}^{\mathrm{T}}{\Sigma'}^{-1}\mathbf{G'})^{-1}\\
\text{Cov}\mathbf{(z')}&= \textcolor{blue}{\boxed{(\mathbf{{G'}}^{\mathrm{T}}{\Sigma'}^{-1}\mathbf{G'})^{-1}}} \\
\end{align*}
\tag{34}
$$

综上，最终可以得到估计的位置坐标 $\mathbf{z}_p$，根据公式 $(25)$ 可以推出：
$$
\begin{split}
\begin{align*}
\mathbf{z'} &= {\left[ \begin{matrix} (x-x_1)^2  \\ (y-y_1)^2   \end{matrix} \right]}   \\
\pm \sqrt{\mathbf{z'}} &= {\left[ \begin{matrix} x-x_1  \\ y-y_1   \end{matrix} \right]}   \\
\end{align*} \\
\textcolor{blue}{\boxed{\mathbf{z}_p = \pm \sqrt{\mathbf{z'}} + {\left[ \begin{matrix} x_1  \\ y_1   \end{matrix} \right]}  }}
\end{split}
\tag{35}
$$
在 $(35)$ 中得到了两个解，解的模糊性可根据有关 MS 的先验信息进行消除。接下来为了找到 $\bold{z}_p$ 的协方差，我们考虑以下的误差表示。
根据 $(24)$ 中的表示，考虑 $\mathbf{z'}$ 和 groundtruth 之间的误差 $\Delta\mathbf{z'}$：
$$
\begin{split}
\begin{align*}
\mathbf{z'}_1-\mathbf{z'}_1^0 &=\mathbf{z'}_1-(x^0-x_1)^2 = 2(x^0-x_1)e_1+e_1^2 \approx  2(x^0-x_1)e_1 \\
\mathbf{z'}_2-\mathbf{z'}_2^0 &=\mathbf{z'}_2-(y^0-y_1)^2= 2(y^0-y_1)e_2+e_2^2 \approx  2(y^0-y_1)e_2 \\
& \Longrightarrow \Delta \mathbf{z'} = 2\left[ \begin{matrix} x^0-x_1&\\&y^0-y_1  \end{matrix} \right]\left[ \begin{matrix} e_1\\e_2  \end{matrix} \right] \\
& \Longrightarrow \Delta \mathbf{z'} =2\mathbf{B''}\Delta  \mathbf{z}_p
\end{align*}
 \end{split} \tag{36}
$$
并利用前文 $(22)$ 、$(32)$ 、 $(27)$ 及 $(34)$ 所推导的关系，$\bold{z}_p$ 的协方差公式为： 
$$
\begin{align*}
\mathbf{z}^0 + e &= \mathbf{z} \Longrightarrow \Delta \mathbf{z}_p  =    \mathbf{z}_p- \mathbf{z^0}\\
\Delta \mathbf{z'} &=2\mathbf{B''}\Delta  \mathbf{z}_p \Longrightarrow \Delta \mathbf{z}_p = \frac{1}{2}(\mathbf{B''}^{T} \Sigma''^{-1} \mathbf{B''})^{-1}\mathbf{B''}^{T} \Sigma''^{-1} \Delta \mathbf{z'} \\
 \Sigma'' &=\text{Cov}(\Delta \mathbf{z'})= \text{Cov}( \mathbf{z'})\\
& \textcolor{red}{\Longrightarrow} \\
\Phi &= \text{Cov}(\bold{z}_p)  \\ 
&= \mathbb{E}(\Delta   \mathbf{z}_p\Delta   \mathbf{z}_p^{\mathrm{T}})   \\
&=  \frac{1}{4} \mathbf{B''}^{-1}  \text{Cov}\mathbf{(z')} \mathbf{B''}^{-1}  \\
&= ( C^2\mathbf{B''}\mathbf{G' }^{T} \mathbf{B'}^{-1}\mathbf{G^0}^{T}\mathbf{B}^{-1}\mathbf{Q}^{-1}\mathbf{B}^{-1}\mathbf{G^0}\mathbf{B'}^{-1}\mathbf{G' } \mathbf{B''})^{-1}
\end{align*}
\tag{37}
$$

### 2.4 总结

以上两次 WLS 计算都假定TDOA测量误差服从零均值的高斯分布, 在这一前提下, 能得到对 MS 位置的最大似然 (ML) 估计, 否则定位误差将会显著增大。接下来我们把以上推导过的思路按照两种情况总结如下。
- 1. 远距离节点的定位公式
     $$
     \begin{align*}
      \widetilde{\mathbf{z}} &\approx (\mathbf{G}^{\mathrm{T}}\mathbf{Q}^{-1}\mathbf{G})^{-1}\mathbf{G}^{\mathrm{T}}\mathbf{Q}^{-1}\mathbf{h} \\
      \textcolor{blue}\Downarrow \\
     \mathbf{z'}&= (\mathbf{G'}^{\mathrm{T}}{\Sigma'}^{-1}\mathbf{G'})^{-1}\mathbf{G'}^{\mathrm{T}}{\Sigma'}^{-1}\mathbf{h'} \\
     &\approx   (\mathbf{G'}^{\mathrm{T}}  \mathbf{B'}^{-1} \mathbf{G}^{\mathrm{T}}\mathbf{Q}^{-1}\mathbf{G} \mathbf{B'}^{-1} \mathbf{G'})^{-1}\mathbf{G'}^{\mathrm{T}}  \mathbf{B'}^{-1} \mathbf{G}^{\mathrm{T}}\mathbf{Q}^{-1}\mathbf{G} \mathbf{B'}^{-1} \mathbf{h'} \\
      \textcolor{blue}\Downarrow \\
     \mathbf{z}_p &= \pm \sqrt{\mathbf{z'}} + {\left[ \begin{matrix} x_1  \\ y_1   \end{matrix} \right]}
     \end{align*}
     $$

- 2. 近距离节点的定位公式
     $$
     \begin{align*}
      \widetilde{\mathbf{z}} &\approx (\mathbf{G}^{\mathrm{T}}\mathbf{Q}^{-1}\mathbf{G})^{-1}\mathbf{G}^{\mathrm{T}}\mathbf{Q}^{-1}\mathbf{h} \quad \Longrightarrow  \quad \mathbf{B}\\
      \textcolor{blue}\Downarrow \\
     \mathbf{z} &= (\mathbf{G}^{\mathrm{T}}W\mathbf{G})^{-1}\mathbf{G}^{\mathrm{T}}W\mathbf{h}\quad \Longrightarrow  \quad \mathbf{B'}\\
      \textcolor{blue}\Downarrow \\
     \mathbf{z'}&= (\mathbf{G'}^{\mathrm{T}}{\Sigma'}^{-1}\mathbf{G'})^{-1}\mathbf{G'}^{\mathrm{T}}{\Sigma'}^{-1}\mathbf{h'} \\
      \textcolor{blue}\Downarrow \\
     \mathbf{z}_p &= \pm \sqrt{\mathbf{z'}} + {\left[ \begin{matrix} x_1  \\ y_1   \end{matrix} \right]}
     \end{align*}
     $$





## 三、克拉美罗界

一些研究领域（如无线定位方向）都会碰到参数估计的问题，这时常常会看到**克拉美罗界** (Cramér–Rao bound)  这个东西。在参数估计和统计中，Cramer-Rao界限（Cramer-Rao bound, **CRB**）或者 Cramer-Rao 下界（**CRLB**），表示一个**确定性参数的估计的方差下界**。命名是为了纪念 Harald Cramer 和 Calyampudi Radhakrishna Rao。这个界限也称为 Cramer-Rao 不等式或信息不等式。 
### 3.1 参数估计问题

一个简单的例子：一个物理量为 $D$，我们使用某种方式去观测它，得到的是观测值为 $x$，由于存在噪声，得到的观测值不是真实值即 $x=D+n$。其中假设 $n\sim(0,\sigma^2)$。

这种情况下我们可以直接用观测值 $x$ 去估计  $D$ ，这样就必然会存在估计误差，直观地理解就是噪声的方差 $\sigma^2$ 越大，估计就可能越不准确。

### 3.2 CRB 的来龙去脉

讨论克拉美罗界就是为了使用这个标准来衡量**无偏估计量**的性能。通俗讲，CRLB 可以用于计算无偏估计中**能够获得的最佳估计精度**，因此经常用于计算理论能达到的最佳估计精度，和评估参数估计方法的性能（是否接近CRLB下界）。 

采用上面的方式，用 $\hat{D}=x$ 去估计 $D$，估计值会在真实值附近波动。克拉美罗界不关心具体的估计方式，只是去反映利用已有信息所能估计参数的**最好效果**。比如上面的例子中当我们观察到 $x$ 的时候，我们可以得到 $x$ 的概率密度函数 PDF 是以 $D$ 为均值，以 $\sigma^2$ 为方差的正态分布：
$$
f(x\ ;D) = \frac{1}{\sqrt{2\pi \sigma^2}}\text{exp}\left[ -\frac{1}{2\sigma^2}(x-D)^2 \right] \tag{38}
$$
当观测到 $x=3$ 的时候，则式 $(38)$ 就转化为 $D$ 的 PDF：
$$
f(D) = \frac{1}{\sqrt{2\pi \sigma^2}}\text{exp}\left[ -\frac{1}{2\sigma^2}(3-D)^2 \right] \tag{39}
$$
假设方差不同的 PDF 如下图所示：

![pdf](E:\Documents\Markdown\pdf.png)

函数图像越尖的话，代表估计精度可以越高。尖锐程度可以用 $1/\sigma^2$ 表示。方差可以描述尖锐程度，但方差只给出了一个估计量 $x=3$ 的信息，如何找到一种对估计精度更好的描述，从而可以反映整个估计问题的信息（所有可能的估计量）？答案就是**似然函数的二阶导数**。

事实上，式 $(39)$ 是一个似然函数，取对数并求导可得：
$$
\begin{split}
\text{ln}f(x[0];\ D) =-\text{ln}{\sqrt{2\pi \sigma^2}}-\frac{1}{2\sigma^2}(x[0]-D)^2 \\
\frac{\partial\  \text{ln}f(x[0];\ D)}{\partial D} = \frac{1}{\sigma^2}(x[0]-D)  \\
\textcolor{blue}{\boxed{\frac{\partial^2\  \text{ln}f(x[0];\ D)}{\partial D^2} =- \frac{1}{\sigma^2} }}
\end{split}
\tag{40}
$$
由式 $(40)$ 继而得到估计量的方差：
$$
\sigma^2 =  \frac{1} {-\frac{\partial^2\  \text{ln}f(x[0];\ D)}{\partial D^2}}  \tag{41}
$$

不失一般性地考虑，如果结果依赖于多个观测值数据 $\bf{x}$ 以及参数 $\theta$，则需要求二阶导数（曲率）更一般的度量即期望。这个期望就是**费雪信息（Fisher）**：
$$
\begin{split}
\textcolor{green}{\boxed{I(\theta) = \mathbb{E}\left[\left(\frac{\partial \ln f(\mathbf{x}; \theta)}{\partial \theta}\right)^2\right] =-\mathbb{E}\left[\frac{\partial^2 \ln f(\mathbf{x}; \theta)}{\partial \theta^2}\right] }}\\
\textcolor{red}{\text{note:} }\quad \mathbb{E}\left[\frac{\partial^2 \ln f(\mathbf{x}; \theta)}{\partial \theta^2}\right] = \int \frac{\partial^2 \ln f(\mathbf{x}; \theta)}{\partial \theta^2}f(x; \theta)dx
\end{split}
\tag{42}
$$
式 $(42)$ 中的期望 $\mathbb{E}(\cdot)$ 都是对于 $f(\mathbf{x};\theta)$ 计算的。在这里我们对上面公式进行详细的说明：

- 正则化条件（似然函数的值应该取到最大值，故令似然函数的一阶导数为 0）
$$
\frac{\partial}{\partial \theta}  \ln f(\mathbf{x}; \theta) = \frac{\frac{\partial}{\partial \theta}  f(\mathbf{x}; \theta)}{ f(\mathbf{x}; \theta)}  \\
\Longrightarrow  \frac{\partial}{\partial \theta} f(\mathbf{x}; \theta) = \frac{\partial}{\partial \theta} \ln  f(\mathbf{x}; \theta) \cdot  f(\mathbf{x}; \theta)\\
\int f(x_i; \theta) dx_i=1 \\
\Longrightarrow    \textcolor{blue}{ \boxed{
\mathbb{E}\left[ \frac{\partial \ln f(\mathbf{x}; \theta)}{\partial \theta} \right ]=\int \frac{\partial \ln f(\mathbf{x}; \theta)}{\partial \theta}\cdot  f(x; \theta)dx=\frac{\partial }{\partial \theta}\int f(\mathbf{x}; \theta)dx=0 }}
$$
- **MLE 的方程的方差**

$$
\begin{align*}
I(\theta) &= \mathbb{E}\left[\left(\frac{\partial \ln f(\mathbf{x}; \theta)}{\partial \theta}\right)^2\right] =  \left[\mathbb{E}\left(\frac{\partial \ln f(\mathbf{x}; \theta)}{\partial \theta}\right)\right]^2+\text{Var}\left[\frac{\partial \ln f(\mathbf{x}; \theta)}{\partial \theta}\right]  \\
&=\text{Var}\left[\frac{\partial \ln f(\mathbf{x}; \theta)}{\partial \theta}\right]   \\
\frac{\partial^2 \ln f(\mathbf{x}; \theta)}{\partial \theta^2}&= \frac{\frac{\partial^2}{\partial \theta^2} f(\mathbf{x}; \theta)}{f(\mathbf{x}; \theta)}-\left[ \frac{\partial}{\partial \theta} \ln  f(\mathbf{x}; \theta)\right]^2  \\
\mathbb{E}\left[\frac{\partial^2 \ln f(\mathbf{x}; \theta)}{\partial \theta^2}\right]&=\int \left\{\frac{\frac{\partial^2}{\partial \theta^2} f(\mathbf{x}; \theta)}{f(\mathbf{x}; \theta)}-\left[ \frac{\partial}{\partial \theta} \ln  f(\mathbf{x}; \theta)\right]^2  \right\}f(x; \theta)dx\\
&=\frac{\partial^2}{\partial \theta^2}\int f(x; \theta)dx - \int \left[ \frac{\partial}{\partial \theta} \ln  f(\mathbf{x}; \theta)\right]^2 f(x; \theta)dx \\
&=-\mathbb{E}\left[\left(\frac{\partial \ln f(\mathbf{x}; \theta)}{\partial \theta}\right)^2\right]=-I(\theta)
\end{align*}
$$

因此，假设 $\theta$ 是一个位置确定性参数,我们需要从观察变量 $x$ 估计它。而它们满足一个概率密度函数 $f(x;θ)$。任何 $\theta$ 的无偏估计 $\hat{\theta}$ 的方差的下界为 Fisher 信息的倒数：
$$
\begin{equation}
\mathrm{Var}({\hat{\theta}}) \geqslant \frac{1}{-\mathbb{E}\left[\frac{\partial^2 \ln f(\mathbf{x}; \theta)}{\partial \theta^2}\right] }
\end{equation}
\tag{43}
$$
其中，右侧的表达式就是 CRLB，它是参数 $\theta$ 的函数。无线定位方法常采用理论下界 CRLB 与定位**解的均方根误差**进行比较，来判断定位估计器的准确率。

## 四、MATLAB 仿真

该部分根据第大**二**节的内容总结，实现其代码。
首先，我们来看 Chan 算法，将其作为一个函数来实现。根据公式我们需要首先设置好各个基站的坐标，以及到达时间差，假定基站的个数 BSN 的范围是：3 < BSN < 7。

```matlab
% 基站数目
BSN = 4;
% 各个基站的位置，2*BSN 的矩阵存储，每一列是一个坐标。
BS = [0, sqrt(3), 0.5*sqrt(3), -0.5*sqrt(3), -sqrt(3), -0.5*sqrt(3), 0.5*sqrt(3); 
      0,      0,         1.5,          1.5,        0,         -1.5,        -1.5]; 
BS = BS(:,1:BSN);
BS = BS .* 50;
% 噪声方差
Noise = 1;
% R=R_{i,1},是加上了噪声后，BSi 与 BS1 到 MS 的距离差，实际中应由 TDOA * c 算得
for i = 1: BSN-1
    R(i) = R0(i+1) - R0(1) + Noise * randn(1); 
end
```

Chan 函数主要包括两部分：第一次 WLS；第二次 WLS。其中 参数需要有 BS 坐标以及到达时间差。因为设置的 BS 的数目可能有所不同，因此为了方便从坐标矩阵中取出特定的点，我们还需要一个 BSN 作为参数。

```matlab
% Zp 为估计的坐标
function Zp = myChan2(BSN, BS, R)
```

第一次 WLS：

```matlab
    % 噪声协方差矩阵
    Q = eye(BSN-1);
    % 第一次 LS
    K1 = 0; % 第一个 BS 坐标设为（0，0）
    for i = 1: BSN-1
        K(i) = BS(1,i+1)^2 + BS(2,i+1)^2;
    end
    % Ga
    for i = 1: BSN-1
        Ga(i,1) = -BS(1, i+1);
        Ga(i,2) = -BS(2, i+1);
        Ga(i,3) = -R(i);
    end
    % h
    for i = 1: BSN-1
        h(i) = 0.5*(R(i)^2 - K(i) + K1);
    end
    % 给出 Z 的初始估计值
    Za0 = inv(Ga'*inv(Q)*Ga)*Ga'*inv(Q)*h';
    % 利用这个粗略估计值计算 B
    B = eye(BSN-1);
    for i = 1: BSN-1
        B(i,i) = sqrt((BS(1,i+1) - Za0(1))^2 + (BS(2,i+1) - Za0(2))^2);
    end
    % FI 
    FI = B*Q*B;
    % 第一次LS结果
    Za1 = inv(Ga'*inv(FI)*Ga)*Ga'*inv(FI)*h';
```

第二次 WLS 的代码实现：

```matlab
    % 第一次 LS 结果的协方差
    CovZa = inv(Ga'*inv(FI)*Ga);
    % 第二次 LS
    sB = eye(3); % sB
    for i = 1: 3
        sB(i,i) = Za1(i);
    end
    % sFI：
    sFI = 4*sB*CovZa*sB;
    % sGa：
    sGa = [1, 0; 0, 1; 1, 1];
    % sh
    sh  = [Za1(1)^2; Za1(2)^2; Za1(3)^2];
    % 第二次 LS 结果
    Za2 = inv(sGa'*inv(sFI)*sGa)*sGa'*inv(sFI)*sh;
    sZ = sqrt(abs(Za2)); 
    % 输出
    if nargout == 1
        Zp = sZ;
    elseif nargout == 0
        disp(sZ);
    end
```

## References

- [1] 来源：知乎. 作者：融冰C .链接：https://zhuanlan.zhihu.com/p/30949000
- [2] 作者：阿毛. 原文链接：[加权最小二乘法与局部加权线性回归](https://uqer.io/v3/community/share/57887c7e228e5b8a03932c66)
- [3] 朱永龙. 基于UWB的室内定位算法研究与应用[D]. 山东大学, 2014. 
- [4] 李万春. 外辐射源定位跟踪技术的研究[D]. 电子科技大学, 2009.
- [5] 李招华, 汪毓铎, 邵青. 基于Chan的TDOA三维定位算法[J]. 现代电信科技, 2014(11):36-40.
- [6] 汪子嘉, 于宏毅, 胡赟鹏. 基于最大似然准则的CHAN改进算法[J]. 计算机应用与软件, 2014(9):240-243.
- [7] 郑飞, 郑继禹. 基于 TDOA 的 CHAN 算法在 UWB 系统 LOS 和 NLOS 环境中的应用研究[J]. 电子技术应用, 2007, 33(11): 110-113.
- [8] 博客园 http://www.cnblogs.com/rubbninja/p/4512765.html 作者：[rubbninja](http://www.cnblogs.com/rubbninja/)

## APPENDIX

---

- 作者： [@icaoys](https://github.com/yashcao)

- 2018 年 06月 15日
- 本人是刚开始接触定位方面的算法和理论，因此本文内容如有不当和错误的问题，或者您有什么疑问，请邮件联系本人，欢迎一起交流、共同进步！



[^1]: Chan, Yiu-Tong, and K. C. Ho. "A simple and efficient estimator for hyperbolic location." *IEEE Transactions on signal processing* 42.8 (1994): 1905-1915.
[^2]: 协方差矩阵：$\text{Var}(X)= \mathbb{E}{ [(X-\bar{X})(X-\bar{X})^{T}}]= \mathbb{E}{ [\Delta X \Delta X^{T}]}\\ \text{Cov}(X,Y)=\mathbb{E}{ [ (X-\bar{X})(Y-\bar{Y})^{T} ]} \\ \text{Cov}(X,X)=\text{Var}(X)\\ \text{Cov}(Z)=\begin{bmatrix} \text{Cov}(X,X) & \text{Cov}(X,Y) \\ \text{Cov}(Y,X) & \text{Cov}(Y,Y) \end{bmatrix} ,\ \ Z=[X ,Y]$
[^3]: 矩阵的乘积之**元素对应乘积**（element-wise product）或者 Hadamard 乘积（Hadamard product），记为 $\mathbf{A}\odot \mathbf{B}$。例如：$\begin{eqnarray} \left[\begin{array}{c} 1 \\ 2 \end{array}\right] \odot \left[\begin{array}{c} 3 \\ 4\end{array} \right] = \left[ \begin{array}{c} 1 * 3 \\ 2 * 4 \end{array} \right] = \left[ \begin{array}{c} 3 \\ 8 \end{array} \right]. \end{eqnarray}$

[^4]: 注意这里是高斯随机变量。**均值为 0 的时候**，自协方差矩阵就是自相关矩阵。 相关矩阵 $R_X=\mathbb{E}(XX^{\mathrm{T}})$。
[^5]: **Gauss-Markov 条件**：$ \left\{ \begin{array} {lr} \mathbb{E}(e_i)=0 \qquad {i=1,2, …,n} \\ \text{Cov} (e_i, e_j)= \left\{ \begin{array}{lr}  \sigma^2  &  & {i=j}\\  0   &  & {i≠j}\\      \end{array} \right.      \end{array} \right.  $
[^6]: 无偏性即 $E(\hat\theta)=\theta$，有效性即 $D(\hat\theta_1)<D(\hat\theta_2) \rightarrow \hat\theta_1 \text{ is more valid than }\hat\theta_2$，一致性即$ \forall\varepsilon> 0, \  \lim\limits_{n\to\infty}P\left(|\hat\theta-\theta|\ge\varepsilon\right)=0$。
[^7]: 矩阵求导用到的定理：$ \frac {\partial tr(AB)} {\partial A} =\frac {\partial tr(BA)} {\partial A}= B^T \\ \frac {\partial tr(A^TB)} {\partial A} =\frac {\partial tr(BA^T)} {\partial A}= B \\ J(\theta) = (X \theta - Y)^T (X \theta - Y)=\text{tr}[(X \theta - Y)^T (X \theta - Y)] \\ \frac {\partial J(\theta)} {\partial \theta} = \frac {\partial {[ \theta^TX^TX \theta- \theta^TX^TY-Y^TX \theta+Y^TY]}} {\partial \theta}= X^TX \theta - X^TY\\ \text{let}\qquad \frac {\partial J(\theta)} {\partial \theta} = X^TX \theta - X^TY=0\\ \theta = (X^TX)^{-1} X^TY$
[^8]: 加权最小二乘法中的**权重缩放**不影响估计结果的原因：$ F_wY=F_wX\beta+F_w\mathbf{e} \Longrightarrow Y^*=X^*\beta+\mathbf{e}^* \Longrightarrow \hat{\beta} \\ \begin{align*} \hat{\beta} & =({X^*}^{\mathrm{T}} {X^*})^{-1} {X^*} ^{\mathrm{T}} {Y^*} \\ &=[(F_wX)^{\mathrm{T}}F_wX]^{-1} (F_wX)^{\mathrm{T}} F_wY \\ &= [X ^{\mathrm{T}} F_w^{\mathrm{T}}F_wX]^{-1} X ^{\mathrm{T}} F_w^{\mathrm{T}} F_wY \\ &= [X ^{\mathrm{T}} WX]^{-1} X ^{\mathrm{T}} WY    \end{align*}  $ 
[^9]: 已知 $\textcolor{red}{\boxed{r_{i,1}=r_{i,1}^0+Cn_{i,1}, \quad  r_{i,1}^0 = r_{i}^0-r_{1}^0}}$，则有 $\begin{align*} \psi &=\mathbf{h} - \mathbf{G z^0} \\ &= \frac{1}{2}[r_{i,1}^2-K_i+K_1] +x_{i,1}x^0+y_{i,1}y^0+r_{i,1}r_1^0 \\&= \frac{1}{2} [(r_{i,1}^0+Cn_{i,1})^2-K_i+K_1] +x_{i,1}x^0+y_{i,1}y^0+( r_{i,1}^0+Cn_{i,1})  r_1^0\\ &= C(r_{i,1}^0+r_1^0)n_{i,1}+\frac{1}{2}C^2n_{i,1}^2\\ &= Cr_i^0n_{i,1}+\frac{1}{2}C^2n_{i,1}^2 \end{align*}$
[^10]: 陆音, 王宝全, 丘觐玮. CHAN 算法在 LOS 和 NLOS 环境下的定位研究[J]. 计算机技术与发展, 2015, 9: 013.
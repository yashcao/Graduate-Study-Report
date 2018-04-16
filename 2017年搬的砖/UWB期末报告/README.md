# My first little paper

## 1 实验说明
- 该实验在MATLAB下进行
- 标准随机信道进行了不同时刻的多次迭代
- 尝试将多时刻数据下训练的模型在将来时刻的译码预测
- 译码失败

## 2 补充实验
- 单一时刻下训练的数据将能很好地预测译码

- 单一时刻下（信道矩阵不变），单一信噪比，最多需要多少数据进行训练，达到的准确率。

- 训练数据（50）

<div align=center>
<img src="https://github.com/yashcao/Graduate-Study-Report/blob/master/2017年搬的砖/UWB期末报告/50data.png" title="50data"/>
</div>


- 训练数据（50）的译码结果 【经历**16**轮迭代训练】


<div align=center>
<img src="https://github.com/yashcao/Graduate-Study-Report/blob/master/2017年搬的砖/UWB期末报告/50acc.png" title="50acc"/>
</div>


- 训练数据（25）【经历**12**轮迭代训练】


<div align=center>
<img src="https://github.com/yashcao/Graduate-Study-Report/blob/master/2017年搬的砖/UWB期末报告/25acc.png" title="25acc"/>
</div>


- 训练数据（20）【经历**23**轮迭代训练】


<div align=center>
<img src="https://github.com/yashcao/Graduate-Study-Report/blob/master/2017年搬的砖/UWB期末报告/20acc.png" title="20acc"/>
</div>


## 3 实验结论

- 每次训练（50）对应的译码正确率为100%较为稳定，（25，20）不稳定大概在90%-100%之间。
- 训练迭代轮数不定，一般在10-30之间。

## 4 启发
### $ 25个比特对应的250\*25 数据就可百分百正确译（当时时刻）码。
### $ 考虑时变信道的方案：
 #### - 需要信道估计吗？
 #### - 单一译码网络训练能否适应信道变化？
 #### - 设计机器学习译码结构要考虑的因素：信道 SNR 和 信道时变

## 虽然没做出什么成果，但这个实验过程和论文我做的都很满意！

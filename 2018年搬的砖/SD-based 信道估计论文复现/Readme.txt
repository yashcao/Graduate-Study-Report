Paper title: Beamspace Channel Estimation for Millimeter-Wave Massive MIMO Systems with Lens Antenna Array

This simulation code package is mainly used to reproduce the results of the following paper:

Linglong Dai, Xinyu Gao, Shuangfeng Han, Chih-Lin I, and Xiaodong Wang, "Beamspace channel estimation for millimeter-wave massive MIMO systems with lens antenna array," in Proc. IEEE ICCC'16, Chengdu, China, Jul. 2016

The author in charge of this simulation code pacakge is: Xinyu Gao (email: xy-gao14@mails.tsinghua.edu.cn).

Reference: We highly respect reproducible research, so we try to provide the simulation codes 
for our published papers (more information can be found at: http://oa.ee.tsinghua.edu.cn/dailinglong/). 
If you in any way use this code for research that results in publications, please cite our original paper 
listed above.

Copyright reserved by the Broadband Communications and Signal Processing Laboratory
(led by Dr. Linglong Dai), Department of Electronic Engineering, Tsinghua University, 
Beijing 100084, China. 

Abstract of Paper: 

By employing the lens antenna array, beamspace
MIMO can utilize beam selection to reduce the number of
required RF chains in mmWave massive MIMO systems without
obvious performance loss. However, to achieve the capacityapproaching
performance, beam selection requires the accurate
information of beamspace channel of large size, which is challenging,
especially when the number of RF chains is limited. To
solve this problem, in this paper we propose a reliable support
detection (SD)-based channel estimation scheme. Specifically, we
propose to decompose the total beamspace channel estimation
problem into a series of sub-problems, each of which only
considers one sparse channel component. For each channel
component, we first reliably detect its support by utilizing the
structural characteristics of mmWave beamspace channel. Then,
the influence of this channel component is removed from the total
beamspace channel estimation problem. After the supports of all
channel components have been detected, the nonzero elements of
the sparse beamspace channel can be estimated with low pilot
overhead. Simulation results show that the proposed SD-based
channel estimation outperforms conventional schemes and enjoys
satisfying accuracy, even in the low SNR region.

How to use this simulation code package?

This package contains Matlab-script that generate Fig. 5 in the paper: main_SD_UPA.m


Enjoy the reproducible research!

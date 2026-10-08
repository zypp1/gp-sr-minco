# GP-SR-MINCO：面向单帧 LiDAR 局部规划的全局提议与共享局部修正

本文为论文第 1—4 节正文草稿，对应当前单 Query、双尺度编码、三层全局提议与三次共享 H10 修正的实现。
关键定义、公式和结构细节均在正文给出；训练目标与实验结果另行成节，不设置附录。

## 1. 引言

未知环境中的无人机局部规划需要将有限视场内的几何观测与当前运动状态相结合，持续生成可供控制器跟踪的连续轨迹。
这一任务包含两个相互耦合的问题：选择与障碍布局相适应的空间运动形态，以及为该形态分配与起始状态相容的时间和边界导数。
仅有几何上合理的路径不足以确定可执行轨迹；较小的位置或时间分配变化，也可能引起明显的加速度与 jerk 变化。

基于优化的方法通过显式轨迹表示和约束处理建立可解释的规划过程。
MINCO 为内部位置、分段时间与连续多项式系数之间提供了解析映射，使低维参数能够控制整条轨迹。
其完整规划框架还通过外层优化处理环境与动力学要求；轨迹表示本身并不等价于约束满足。
这一分工为学习式轨迹生成提供了结构化接口。[Wang 等，2022](https://arxiv.org/abs/2103.00190)

学习型方法则将观测到轨迹的映射交由神经网络实现。
YOPO 通过预测运动基元的偏移与评分完成单阶段规划，NEO-Planner 利用网络预测初始化后续空间—时间优化。
两者分别体现了直接生成与学习辅助优化的不同分工。
[Lu 等，2024](https://wayhome316.github.io/projects/onlyplanonce/youonlyplanonce.pdf)，
[Chen 等，2025](https://arxiv.org/abs/2309.10683)

本文关注单帧 LiDAR 条件下、不调用在线非线性轨迹优化器的局部轨迹生成。
在这一设定中，环境交互需要兼顾全视场结构与候选轨迹邻域：前者支持整体运动提议，后者为具体轨迹修正提供几何依据。
与此同时，若每次局部查询都依赖精确轨迹解码，中间参数更新将与连续轨迹求解反复耦合。
因此，我们将环境特征的查询几何与最终执行轨迹的解析生成分离。

为此，本文构建 Global Proposal–Shared Refiner MINCO 规划器，简称 GP-SR-MINCO。
网络首先利用双尺度 LiDAR 特征和联合状态—轨迹参数 Query 生成全局提议；随后由同一 Refiner 连续修正三次。
每次修正均根据当前参数重新构造 Hermite 查询代理，并从精细特征图读取局部环境。
最终参数通过 minimum-jerk MINCO 解码为满足给定起终边界的三段连续轨迹。
训练时在网络外解码各阶段输出，以真实物理时间上的专家轨迹提供监督；推理时仅解码最终输出。

本文的方法特点体现在以下三个方面：

1. 将完整运动状态与完整轨迹参数编码为单个联合 Query，以粗尺度全视场交互生成提议，以精细局部交互完成修正。
2. 使用随当前参数更新的 Hermite 查询代理组织局部几何信息，并通过参数共享的修正模块联合更新空间参数、终端导数与时间比例。
3. 将学习式参数更新与精确 MINCO 解码解耦，在保留连续轨迹监督的同时，使网络推理不依赖中间阶段的 MINCO 求解。

上述设计不引入候选库、历史扫描或上一帧轨迹输入。
本文所称“全局”仅指当前扫描的完整视场，而非完整地图；最小积分 jerk、共享修正和轨迹模仿也不构成无碰撞、峰值动力学合格或跨帧一致性的保证。

## 2. 相关工作

### 2.1 学习型无人机局部规划

YOPO 将局部规划表示为一组空间分布的多项式运动基元及其评分预测，通过基元偏移调整轨迹，并利用训练阶段的环境代价梯度指导学习。
这一方法说明，学习式规划可以直接面向轨迹生成，而不必在推理阶段完整复现传统搜索—优化流程。
本文同样采用观测驱动的轨迹生成，但只输出一条联合参数化轨迹，不设置基元候选评分与选择分支。
[Lu 等，2024](https://wayhome316.github.io/projects/onlyplanonce/youonlyplanonce.pdf)

NEO-Planner 从传感器观测预测 MINCO 空间与时间参数，再以该预测初始化在线轨迹优化。
它与本文共享结构化轨迹参数接口，但网络输出的角色不同：NEO-Planner 提供优化初值，本文的最终参数直接进入解析解码器，不经过外层空间—时间优化。
因此，不能将其后端优化所提供的约束处理能力视为本文已有能力。
[Chen 等，2025](https://arxiv.org/abs/2309.10683)

近期预印本 Neural-Primitive 使用基于离线运动基元的模仿学习，直接从感知输入预测多项式系数。
它代表了另一种避免在线后端求解的路线。
与直接系数回归不同，本文学习内部位置、终端边界与分段比例，由 MINCO 恢复系数；两者的差异在于网络承担的轨迹自由度及边界条件的实现方式，而非是否使用神经网络生成轨迹。
[Liu 等，2026，预印本](https://arxiv.org/abs/2608.20948)

### 2.2 结构化轨迹表示与可微生成

MINCO 使用稀疏空间—时间参数表示满足给定边界和内部约束的最小控制代价轨迹，并支持对这些参数求导。
本文采用其中三阶积分链对应的 minimum-jerk 表示，将其作为网络输出到连续轨迹的解析层，而不采用完整的几何约束优化框架。
[Wang 等，2022](https://arxiv.org/abs/2103.00190)

AllocNet 从安全飞行走廊与边界状态学习时间分配，随后通过带约束的二次规划生成轨迹，并利用隐式优化层训练网络。
其方法突出时间分配对轨迹可行性的作用。
本文同样保留可学习时间，但固定总时长，并联合预测空间参数和终端导数；解析器不接收安全走廊，也不求解包含碰撞不等式的 QP。
[Wu 等，2024](https://arxiv.org/abs/2309.15191)

Optimization-Embedded Networks 从深度观测提取空间约束，再将其送入可微的模型驱动空间—时间优化过程。
其学习对象与本文不同：前者学习优化器所需的约束，本文学习确定轨迹的参数。
这一差异使两者在约束满足机制和在线计算组成上不能直接等同。
[Han 等，2025](https://zju-fast-lab.github.io/e2e_opt/)

上述工作表明，学习与解析结构可以在初值、约束、时间或完整轨迹参数等不同层面结合。
本文选择的分工是：网络负责观测条件下的空间—时间参数决策，MINCO 负责给定参数下的连续轨迹恢复。
这一分工提供明确的物理接口，但不将环境安全性转化为解析保证。

### 2.3 面向规划的特征交互与轨迹修正

UniAD 通过统一 Query 接口组织感知、预测与规划任务，强调上游特征应服务于最终规划目标。
本文借鉴其任务导向的交互思路，但不构建检测、跟踪、地图与多参与者预测的完整多任务系统，而由轨迹 Query 直接读取 LiDAR 特征。
[Hu 等，2023](https://openaccess.thecvf.com/content/CVPR2023/html/Hu_Planning-Oriented_Autonomous_Driving_CVPR_2023_paper.html)

MTR 使用运动 Query 对多模态未来运动进行提议与修正，并根据已预测轨迹动态收集邻近地图元素。
这一机制说明，当前轨迹可以作为下一阶段环境读取的条件。
本文将该原则用于无人机自身的三维局部规划，不使用多模态意图 Query，也不将交通参与者预测指标作为规划安全指标。
[Shi 等，2022](https://proceedings.neurips.cc/paper_files/paper/2022/file/2ab47c960bfee4f86dfc362f26ad066a-Paper-Conference.pdf)

SmartRefine 进一步采用锚点中心化的上下文编码，依据运动速度与修正阶段调整检索范围，并通过质量评分决定修正次数。
本文同样围绕当前轨迹组织局部关系，但采用固定三次共享修正、预设的递减角窗口与联合参数输出，不引入质量评分、自适应停止或分段循环输出机制。
[Zhou 等，2024](https://arxiv.org/abs/2403.11492)

因此，本文并非重新提出 cross-attention、参数共享或轨迹条件化查询，而是研究这些机制在单帧 LiDAR—MINCO 接口下的具体组织方式。
关键区别在于：局部上下文由无需 MINCO 求解的代理曲线确定，修正作用于完整轨迹参数，最终连续轨迹仍由精确解析层生成。

## 3. 问题定义与预备知识

### 3.1 输入、坐标系与任务范围

令当前同步运动状态的时间为原点，固定规划时域为 $H=2\,\mathrm{s}$。
输入为一帧归一化 LiDAR range image $I\in[0,1]^{1\times30\times180}$，以及

$$
\boldsymbol{s}^{\mathcal B}
=\left[\boldsymbol{v}_0^{\mathcal B},\boldsymbol{a}_0^{\mathcal B},\boldsymbol{g}^{\mathcal B}\right]
\in\mathbb R^9.
\tag{1}
$$

其中，$\mathcal B$ 为当前真实机体系，$\boldsymbol g^{\mathcal B}$ 为相对目标位置，起点为机体系原点。
设 $R_{\mathcal W\mathcal B}$ 将机体系向量旋转到世界系，$\boldsymbol x_0^{\mathcal W}$ 为当前位置，则

$$
\boldsymbol x^{\mathcal B}
=R_{\mathcal W\mathcal B}^{\mathsf T}(\boldsymbol x^{\mathcal W}-\boldsymbol x_0^{\mathcal W}),
\qquad
\boldsymbol v^{\mathcal B}=R_{\mathcal W\mathcal B}^{\mathsf T}\boldsymbol v^{\mathcal W},
\qquad
\boldsymbol a^{\mathcal B}=R_{\mathcal W\mathcal B}^{\mathsf T}\boldsymbol a^{\mathcal W}.
\tag{2}
$$

$R_{\mathcal W\mathcal B}$ 由完整四元数得到，不省略 roll 与 pitch。
LiDAR 帧与最近真实 odometry 行配对，状态和未来标签使用同一个 odometry 时间原点。
网络不接收历史观测、上一计划、全局地图或障碍物语义类别。
任务输出是面向目标的局部轨迹 $\boldsymbol p:[0,H]\rightarrow\mathbb R^3$，并不要求 $\boldsymbol p(H)=\boldsymbol g$。

输入扫描的最大量程为 $r_{\max}=20\,\mathrm m$，水平 $180$ 列覆盖 $360^\circ$；
垂直方向保留 $[-7^\circ,52^\circ]$ 内的 $30$ 个射线中心。
每格保存最近返回距离除以 $r_{\max}$，值为 $1$ 表示量程内无返回。
无返回不代表视场外或量程外无障碍，单帧输入也不提供对隐藏区域的完整描述。

### 3.2 轨迹参数与时间分配

轨迹由 $M=3$ 段组成，内部位置点为 $\boldsymbol Q_1,\boldsymbol Q_2$，终端边界为
$\boldsymbol p_e,\boldsymbol v_e,\boldsymbol a_e$。
网络使用以下 $18$ 维参数向量：

$$
\boldsymbol z=
\left[\bar{\boldsymbol Q}_1,\bar{\boldsymbol Q}_2,
\bar{\boldsymbol p}_e,\bar{\boldsymbol v}_e,\bar{\boldsymbol a}_e,
\boldsymbol\ell\right],
\qquad
\boldsymbol r=\operatorname{softmax}(\boldsymbol\ell),
\qquad T_i=H r_i.
\tag{3}
$$

上划线表示按相应物理尺度归一化，$\boldsymbol\ell\in\mathbb R^3$ 为时间 logits。
因而 $T_i>0$ 且 $\sum_iT_i=H$，但三段时长不必相等。
内部位置、终端边界和时间比例共同确定曲线；$\boldsymbol Q_i$ 是分段连接处的插值位置，并非 B 样条控制点。

定义累计时刻 $\tau_0=0$、$\tau_i=\sum_{j=1}^{i}T_j$。
第 $i$ 段以局部时间 $u=t-\tau_{i-1}$ 表示为

$$
\boldsymbol p_i(u)=\sum_{n=0}^{5}\boldsymbol c_{i,n}u^n,
\qquad 0\leq u\leq T_i.
\tag{4}
$$

两个内部点与终端边界共同控制三段五次曲线的空间形态，各段可以表达连续弯曲的几何。
时间比例也不是独立于几何的附属输出：在固定归一化段形状的条件下，$n$ 阶时间导数按 $T_i^{-n}$ 缩放；
当物理边界导数保持不变时，改变时长还会改变多项式形状。
本文由联合参数 Query 与联合输出头处理这些耦合关系，不采用先固定几何再独立分配时间的网络分支。

### 3.3 Minimum-jerk MINCO 解码

对给定的内部位置、正时长和起终边界，解析层恢复满足约束的最小积分 jerk 轨迹：

$$
\begin{aligned}
\min_{\boldsymbol p}\quad&\sum_{i=1}^{3}\int_0^{T_i}\|\boldsymbol p_i^{(3)}(u)\|_2^2\,\mathrm du,\\
\text{s.t.}\quad&\left(\boldsymbol p_1,\dot{\boldsymbol p}_1,\ddot{\boldsymbol p}_1\right)(0)
=\left(\boldsymbol 0,\boldsymbol v_0,\boldsymbol a_0\right),\\
&\left(\boldsymbol p_3,\dot{\boldsymbol p}_3,\ddot{\boldsymbol p}_3\right)(T_3)
=\left(\boldsymbol p_e,\boldsymbol v_e,\boldsymbol a_e\right),\\
&\boldsymbol p_i(T_i)=\boldsymbol p_{i+1}(0)=\boldsymbol Q_i,\quad i=1,2,\\
&\boldsymbol p_i^{(n)}(T_i)=\boldsymbol p_{i+1}^{(n)}(0),\quad n=1,2,\ i=1,2.
\end{aligned}
\tag{5}
$$

式（5）是既有 MINCO 最优性条件的三阶积分链特例。
段内 Euler–Lagrange 条件给出 $\boldsymbol p_i^{(6)}=0$，因此解为分段五次多项式。
对内部节点，位置固定而速度、加速度可自由变化；对目标泛函分部积分后的边界项要求 jerk 与 snap 同时连续。
由此得到位置至四阶导数连续的连接条件。
[Wang 等，2022，第 IV 节](https://arxiv.org/html/2103.00190v4)

具体到 $M=3$，每个坐标有 $18$ 个多项式系数。
起终 P/V/A 提供 $6$ 个条件，每个内部节点提供左右位置插值和一至四阶导数连续共 $6$ 个条件，合计 $18$ 个。
将三个坐标的系数并列，得到

$$
A(\boldsymbol T)C
=Y(\boldsymbol Q_1,\boldsymbol Q_2,\boldsymbol p_e,\boldsymbol v_e,\boldsymbol a_e,
\boldsymbol v_0,\boldsymbol a_0),
\qquad A\in\mathbb R^{18\times18},\quad C,Y\in\mathbb R^{18\times3}.
\tag{6}
$$

当前实现直接求解这一固定规模线性系统，内部采用 float64，CUDA 使用 cuSOLVER，求解后恢复输入数值类型。
其可微关系可写为 $\mathrm dC=A^{-1}(\mathrm dY-\mathrm dA\,C)$；实现通过线性求解及自动微分计算，不显式构造逆矩阵。
因此，轨迹上的监督可以同时更新几何、终端导数与时间参数。
本文不将这一固定规模实现描述为通用段数下的线性复杂度求解器。

最小积分 jerk 是给定参数后的曲线性质，并非对网络参数执行避障优化。
它保证所定义的插值与连续性条件，不保证峰值 jerk 不超限，也不包含碰撞、推力、姿态或执行器约束。
尤其是单条曲线内部连续，不意味着相邻规划帧预测的两条曲线在重叠时域内一致。

## 4. 模型结构

### 4.1 总体计算流程

GP-SR-MINCO 由双尺度环境编码器、全局提议模块、共享局部修正模块和外部解析解码器组成。
环境编码器只运行一次，产生粗尺度记忆 $E_c$ 与精细尺度记忆 $E_f$；状态编码器也只运行一次，得到共享状态上下文 $\boldsymbol c_s$。
全局模块以恒速参考参数 $\boldsymbol z_{\mathrm{ref}}$ 构造 Query，读取 $E_c$ 后输出 $\boldsymbol z_0$。
随后进行三次局部修正，得到 $\boldsymbol z_1,\boldsymbol z_2,\boldsymbol z_3$，最后由式（6）解码 $\boldsymbol z_3$。

| 组件 | 输入与连接 | 输出维度 |
| --- | --- | --- |
| 精细主干 | 5 通道观测，$3\times3$ stem，两个 64 通道残差块 | $B\times64\times30\times180$ |
| 粗尺度主干 | 精细主干经 $2\times2$ 平均池化，$64\rightarrow128$ 投影，三个残差块 | $B\times128\times15\times90$ |
| 全局环境记忆 | 粗特征投影与射线／观测编码相加 | $B\times1350\times256$ |
| 局部环境记忆 | 精细特征与上采样粗特征融合，再加入射线／观测编码 | $B\times5400\times256$ |
| 状态编码器 | $9\rightarrow128\rightarrow128$ MLP | $B\times128$ |
| 参数编码器 | $18\rightarrow128\rightarrow128$ MLP | $B\times128$ |
| 联合 Query | 参数特征与状态特征拼接，再 LayerNorm | $B\times1\times256$ |
| 全局提议 | 三个独立的 8-head cross-attention + FFN 层，最后联合读出 | $\boldsymbol z_0\in\mathbb R^{B\times18}$ |
| 局部修正 | 同一个 8-head Refiner 调用三次，每次重建 Query 和局部支持 | $\boldsymbol z_1,\boldsymbol z_2,\boldsymbol z_3$ |
| 最终解析层 | $\boldsymbol z_3$ 与真实起始 P/V/A | $B\times3\times6\times3$ 多项式系数 |

表中的三层全局模块只在最后一层后输出一次参数提议，不产生三个独立候选。
三次局部修正共享注意力、FFN、几何编码和输出头，阶段差异由可学习的阶段向量和预设窗口提供。

### 4.2 状态规范化与双尺度环境编码

#### 4.2.1 内部坐标规范化

在真实机体系上进一步构造目标朝向的内部坐标系 $\mathcal C$。
令 $\psi$ 为相对目标在机体 $xy$ 平面的方位角；当该投影退化时，依次使用速度、加速度的平面方向，再退化到机体 $+x$ 轴。
所有向量通过 $R_z(-\psi)$ 转换到 $\mathcal C$，状态归一化为

$$
\bar{\boldsymbol s}
=\left[\boldsymbol v_0^{\mathcal C}/s_v,
\boldsymbol a_0^{\mathcal C}/s_a,
\boldsymbol g^{\mathcal C}/s_g\right],
\qquad (s_v,s_a,s_g)=(6\,\mathrm{m/s},6\,\mathrm{m/s^2},30\,\mathrm m).
\tag{7}
$$

状态归一化不裁剪物理输入。
range image 采用整数列循环移位而不插值，移位量为
$m=\operatorname{round}(180\psi/(2\pi))$，即 $I^{\mathcal C}_{h,w}=I^{\mathcal B}_{h,(w+m)\bmod180}$。
向量连续旋转与扫描整数移位之间保留至多半个水平格的角度量化差异；这不是任意角度下的精确等变性构造。
此后网络内的几何量均在 $\mathcal C$ 中表达，最终输出再旋回 $\mathcal B$。
内部方位规范化不改变式（2）中完整三维世界／机体转换的要求。

#### 4.2.2 共享主干与跨尺度融合

对每条射线 $n$，记归一化距离为 $\rho_n$，单位方向为 $\boldsymbol d_n$，返回标记为 $h_n=\mathbb 1[\rho_n<1]$。
编码器输入不是单独的距离值，而是以下五维观测：

$$
\boldsymbol o_n=
\left[h_n(1-\rho_n),\ h_n\rho_n\boldsymbol d_n,\ h_n\right]\in\mathbb R^5.
\tag{8}
$$

其中距离接近度、归一化返回坐标与返回标记分别表达近障程度、三维几何和测量状态。
无返回时这五维输入为零，但后续方向编码仍保留射线朝向。

精细主干以 $3\times3$ 卷积将输入映射为 $64$ 通道，再经过两个同分辨率残差块，得到 $F_f$。
每个残差块采用“卷积—GroupNorm—SiLU—卷积—GroupNorm—残差相加—SiLU”。
水平使用循环 padding，垂直使用零 padding；GroupNorm 使用 8 组且不含仿射参数。
随后对 $F_f$ 做 $2\times2$ 平均池化，经 $1\times1$ 投影、GroupNorm 与 SiLU 升至 $128$ 通道，
再经过三个残差块，得到 $F_c$。

全局内容特征由 $F_c$ 投影至 $256$ 维；局部内容特征由细、粗特征在 $64$ 通道空间相加融合：

$$
\begin{aligned}
C_c&=P_c(F_c),\\
C_f&=P_o\!\left(\operatorname{SiLU}\!\left(\operatorname{GN}\!\left(
K_{3\times3}\left[P_f(F_f)+\operatorname{Up}(P_m(F_c))\right]\right)\right)\right).
\end{aligned}
\tag{9}
$$

$P_f:64\rightarrow64$、$P_m:128\rightarrow64$、$P_o:64\rightarrow256$ 与 $P_c:128\rightarrow256$
均为 $1\times1$ 卷积，$\operatorname{Up}$ 为最近邻上采样。
该连接使全局与局部分支共享同一感知主干，同时保留精细分支不经池化的路径。
粗尺度用于降低全视场交互的 token 数，精细尺度为局部修正保留原生射线布局；两者并不要求特征逐项一致。
池化存在信息损失，跨尺度融合不能被解释为无损重建。

#### 4.2.3 射线与观测编码

将内容特征展平后，每个环境 token 定义为

$$
\boldsymbol e_n=\boldsymbol c_n+W_d\boldsymbol d_n+E_o(\boldsymbol o_n),
\qquad W_d:3\rightarrow256,\quad E_o:5\rightarrow64\rightarrow256.
\tag{10}
$$

$E_o$ 的中间激活为 SiLU，方向与观测映射均不含偏置，并在两个尺度间共享。
三项分别承担感知内容、确定性射线位置与当前测量状态的表达；同维相加使其进入同一 token 空间，并不预设各项贡献相等。
这不是依赖序号的纯位置编码，range image 中每个 token 都与一条具体物理射线关联。

每个粗格对应 $2\times2$ 原生射线：若存在返回，以其中最近返回的距离与方向作为粗格的几何代表；
若四条射线均无返回，则使用归一化平均方向和最大量程。
特征平均池化与最近返回选择分别作用于 CNN 内容和射线几何属性。
精细记忆保留全部 $5400$ 条原生射线，全局记忆包含 $1350$ 个粗 token。
无返回 token 不从注意力中移除，其方向与测量状态仍可参与环境交互。

### 4.3 联合 Query 与全局轨迹提议

#### 4.3.1 恒速参考与联合参数表达

全局提议从当前速度构造参考参数：

$$
\boldsymbol Q_1^{\mathrm{ref}}=\frac H3\boldsymbol v_0,
\quad\boldsymbol Q_2^{\mathrm{ref}}=\frac{2H}3\boldsymbol v_0,
\quad\boldsymbol p_e^{\mathrm{ref}}=H\boldsymbol v_0,
\quad\boldsymbol v_e^{\mathrm{ref}}=\boldsymbol v_0,
\quad\boldsymbol a_e^{\mathrm{ref}}=\boldsymbol0,
\quad\boldsymbol\ell^{\mathrm{ref}}=\boldsymbol0.
\tag{11}
$$

位置按 $s_p$、终端速度按 $s_{ve}$、终端加速度按 $s_a$ 归一化。
$s_p$ 与 $s_{ve}$ 仅由所选训练标签拟合，分别取最大位置范数、最大终端速度范数的 $1.05$ 倍，下限分别为 $1\,\mathrm m$ 和 $1\,\mathrm{m/s}$；
确定后与模型一起保存，验证和推理不重新估计。
构造参考时，前十二个归一化分量裁剪到 $[-1+10^{-5},1-10^{-5}]$。
参考只是参数更新的起点，不是专家标签；当当前加速度非零时，它也不是满足真实起始加速度的执行轨迹。
当前加速度仍完整进入状态编码器与最终 MINCO 边界。

完整状态和完整参数分别映射到 $128$ 维：

$$
\begin{aligned}
\boldsymbol c_s&=E_s(\bar{\boldsymbol s}),\qquad E_s:9\rightarrow128\rightarrow128,\\
\boldsymbol c_z&=E_z\!\left([\bar{\boldsymbol Q}_1,\bar{\boldsymbol Q}_2,
\bar{\boldsymbol p}_e,\bar{\boldsymbol v}_e,\bar{\boldsymbol a}_e,
\operatorname{softmax}(\boldsymbol\ell)]\right),\quad E_z:18\rightarrow128\rightarrow128,\\
\boldsymbol q(\boldsymbol z,\boldsymbol s)&=\operatorname{LN}([\boldsymbol c_z,\boldsymbol c_s])
\in\mathbb R^{256}.
\end{aligned}
\tag{12}
$$

两个 MLP 均在两层线性映射之间使用 SiLU，并在末端使用 LayerNorm。
状态编码只计算一次；参数编码在各阶段共享权重，但随当前参数重新计算。
参数编码器读取 softmax 比例而非无界 logits，从而消除共同 logit 偏移对 Query 内容的影响。

联合 Query 允许网络在进入注意力前建立位置、边界导数和时间之间的关系，无需为各参数人为分配独立 Query。
它表示一条完整轨迹，不承担多候选身份编码。
由于这里只有一个 token，Query self-attention 的归一化权重会退化为 $1$，不能建立额外的 token 间关系；
因此本文保留 cross-attention 与 FFN，不设置 Query self-attention。

#### 4.3.2 全局相对几何与三层交互

仅有射线位置尚不能直接表达障碍与当前运动趋势的关系。
因此，全局记忆额外编码返回点相对于恒速参考的几何关系。
在 $t_j=jH/20$、$j=1,\ldots,20$ 处取 $\boldsymbol p_j^{\mathrm{ref}}=t_j\boldsymbol v_0$，
这些位置使用未裁剪物理速度直接计算，不调用 MINCO。
对粗格返回点 $\boldsymbol x_n$，记最近参考点索引为 $j_n^*$、位移为
$\boldsymbol\Delta_n=\boldsymbol x_n-\boldsymbol p_{j_n^*}^{\mathrm{ref}}$，距离为 $d_n=\|\boldsymbol\Delta_n\|_2$，构造

$$
\boldsymbol\eta_n=
\left[
\operatorname{clip}(\boldsymbol\Delta_n/r_{\max},-2,2),
\operatorname{clip}(d_n/r_{\max},0,2),
\tanh\!\left(\frac{d_n-r_s}{r_s}\right),
\frac{j_n^*-1}{19}
\right]\in\mathbb R^6,
\qquad r_s=0.3\,\mathrm m.
\tag{13}
$$

最后一项是有序参考索引，净空项是相对于返回点的距离余量，不是 ESDF。
经共享 $6\rightarrow64$ 隐层与两个独立 $64\rightarrow256$ 投影，得到

$$
\boldsymbol k_n^{\mathrm{mem}}=\boldsymbol e_n+h_nW_K^{r}\operatorname{SiLU}(W_r\boldsymbol\eta_n),
\qquad
\boldsymbol v_n^{\mathrm{mem}}=\boldsymbol e_n+h_nW_V^{r}\operatorname{SiLU}(W_r\boldsymbol\eta_n).
\tag{14}
$$

返回标记 $h_n$ 仅关闭无返回射线的相对障碍编码，不关闭环境 token。
$W_K^r,W_V^r$ 初始化为零，使相对分支从零附加量开始学习。

以 $\boldsymbol q^0=\boldsymbol q(\boldsymbol z_{\mathrm{ref}},\boldsymbol s)$ 为初始 Query，
三个参数独立的全局层依次更新

$$
\begin{aligned}
\boldsymbol u^l&=\boldsymbol q^{l-1}+
\operatorname{MHA}_l\!\left(\operatorname{LN}_Q^l(\boldsymbol q^{l-1}),
\operatorname{LN}_K^l(K^{\mathrm{mem}}),\operatorname{LN}_V^l(V^{\mathrm{mem}})\right),\\
\boldsymbol q^l&=\boldsymbol u^l+W_2^l\operatorname{SiLU}\!\left(W_1^l\operatorname{LN}_F^l(\boldsymbol u^l)\right),
\qquad l=1,2,3.
\end{aligned}
\tag{15}
$$

每层包含 8 个注意力头，每头维度 $32$；FFN 由两个线性映射构成，维度为 $256\rightarrow1024\rightarrow256$。
Q/K/V 分别归一化，FFN 使用无仿射参数的 pre-LayerNorm；当前 dropout 为零。
三层读取相同的 $1350$ 个粗记忆 token，但各自使用独立投影更新 Query，不执行环境 token 之间的全局 self-attention。
最后通过 $256\rightarrow256\rightarrow18$ 的联合输出头产生修正量，并按第 4.5 节的有界映射得到 $\boldsymbol z_0$。
全局层内传递 Query 内容，而局部修正阶段将重新由当前参数构造 Query，两者的更新方式不同。

### 4.4 基于 Hermite 查询代理的共享局部修正

#### 4.4.1 无 MINCO 求解的查询几何

局部修正需要知道当前轨迹经过哪些观测区域，但不必在每次读取环境前恢复精确的 MINCO 系数。
本文由当前参数构造分段三次 Hermite 代理，仅用于确定局部窗口中心和切向。
对第 $k$ 次修正的输入 $\boldsymbol z_k$，令
$\boldsymbol P_0=\boldsymbol0$、$\boldsymbol P_1=\boldsymbol Q_1$、$\boldsymbol P_2=\boldsymbol Q_2$、$\boldsymbol P_3=\boldsymbol p_e$，
并由式（3）得到时长 $T_1,T_2,T_3$。
代理的节点速度取为

$$
\boldsymbol\nu_0=\boldsymbol v_0,
\quad\boldsymbol\nu_1=\frac{\boldsymbol P_2-\boldsymbol P_0}{T_1+T_2},
\quad\boldsymbol\nu_2=\frac{\boldsymbol P_3-\boldsymbol P_1}{T_2+T_3},
\quad\boldsymbol\nu_3=\boldsymbol v_e.
\tag{16}
$$

内部速度是相邻两段弦速度按时长加权的平均，具有一致的物理量纲；相邻段共用该速度，使代理在节点处一阶连续。
第 $i$ 段令 $u=(t-\tau_{i-1})/T_i$，则

$$
\begin{aligned}
\tilde{\boldsymbol p}_i(u)
={}&(2u^3-3u^2+1)\boldsymbol P_{i-1}
 +(u^3-2u^2+u)T_i\boldsymbol\nu_{i-1}\\
&+(-2u^3+3u^2)\boldsymbol P_i
 +(u^3-u^2)T_i\boldsymbol\nu_i.
\end{aligned}
\tag{17}
$$

切向按物理时间求导，即 $\dot{\tilde{\boldsymbol p}}_i=T_i^{-1}\partial\tilde{\boldsymbol p}_i/\partial u$。
在十个固定物理时刻 $t_j=jH/10$、$j=1,\ldots,10$ 取样；样本所属的分段由预测累计时长决定，并非每段分配相同数量的点。
每个位置的描述符为

$$
\boldsymbol d_j^{p}=
\left[\tilde{\boldsymbol p}(t_j)/s_p,
\frac{\dot{\tilde{\boldsymbol p}}(t_j)}{\max(\|\dot{\tilde{\boldsymbol p}}(t_j)\|_2,10^{-6})},
t_j/H\right]\in\mathbb R^7.
\tag{18}
$$

Hermite 代理使用起终速度，但不施加起终加速度，也不满足 MINCO 的最小积分 jerk 条件。
其与最终曲线的位置和方向偏差需相对于传感器分辨率评估，并不存在由 $2^\circ$ 水平采样间隔直接给出的代理误差上界。
选择该代理的依据是它以显式多项式运算提供依赖当前几何、时长与边界速度的连续查询位置；
近似是否足以覆盖关键障碍，应由与精确查询的对照验证。
完整状态与 action Query 仍包含起始和终端加速度，代理的简化不等于删除这些网络输入或输出。

#### 4.4.2 原生射线网格上的局部支持

将查询位置的方向投影到原生 range image，水平方向取对应角度格，垂直方向取最近射线行。
位置范数低于 $10^{-4}$ 时改用切向方向；切向也退化时使用内部坐标系的 $+x$ 方向。
围绕每个投影位置，三次调用分别读取完整的 $9\times17$、$5\times9$ 与 $3\times5$ 窗口：

| 修正 | 每个位置的窗口 | 十个位置的最大键／值对数量 |
| --- | ---: | ---: |
| $\boldsymbol z_0\rightarrow\boldsymbol z_1$ | $9\times17$ | 1530 |
| $\boldsymbol z_1\rightarrow\boldsymbol z_2$ | $5\times9$ | 450 |
| $\boldsymbol z_2\rightarrow\boldsymbol z_3$ | $3\times5$ | 150 |

水平索引循环回绕，超出有效垂直行的格子被掩蔽。
若查询中心本身在垂直视场之外，读取最近的边缘行附近窗口，而不是外推未知环境。
窗口大小定义在角度网格上，对应的物理覆盖随距离变化；它不是固定米制半径的安全管道。
不同位置可能读取同一原生射线，此时保留不同的“查询位置—射线”关系，不去重、不做 TopK 筛选。

递减窗口体现由较宽上下文转向局部细化的设计意图，而非假定后一阶段必然更准确。
共享模块的参数维度与窗口中的 token 数无关，因此不同窗口可以使用同一注意力权重。
这里共享的是网络参数，不是预计算的注意力概率；每次调用都重新生成支持、关系特征和注意力分布。

#### 4.4.3 位置条件与相对几何消息

对查询位置 $j$ 与窗口射线 $n$，先构造位置条件化 token

$$
\hat{\boldsymbol e}_{jn}
=\boldsymbol e_n+E_p(\boldsymbol d_j^p)+\boldsymbol b_j,
\qquad E_p:7\rightarrow64\rightarrow256,
\quad\boldsymbol b_j\in\mathbb R^{256}.
\tag{19}
$$

$E_p$ 使用 SiLU，$\boldsymbol b_j$ 是十个有序采样位置的可学习角色向量，初始化为零并在修正阶段间共享。
该编码使同一射线在不同未来位置附近具有不同含义。

令 $\boldsymbol x_n=r_{\max}\rho_n\boldsymbol d_n$，
$\boldsymbol\Delta_{jn}=\boldsymbol x_n-\tilde{\boldsymbol p}_j$，
$\delta_{jn}=\|\boldsymbol\Delta_{jn}\|_2$，$\hat{\boldsymbol t}_j$ 为代理单位切向。
对于未被垂直掩蔽的格子，定义八维距离／窗口特征与四维方向特征：

$$
\begin{aligned}
\boldsymbol\xi_{jn}
&=\left[
h_n\boldsymbol\Delta_{jn}/r_{\max},
\frac{\|\boldsymbol x_n\|_2-\|\tilde{\boldsymbol p}_j\|_2}{r_{\max}},
\frac{\Delta h}{4},\frac{\Delta w}{8},\ h_n,
h_n\tanh\!\left(\frac{\delta_{jn}-r_s}{r_s}\right)
\right]\in\mathbb R^8,\\
\boldsymbol\zeta_{jn}
&=h_n\left[
\hat{\boldsymbol\Delta}_{jn},
\hat{\boldsymbol\Delta}_{jn}^{\mathsf T}\hat{\boldsymbol t}_j
\right]\in\mathbb R^4,
\qquad
\hat{\boldsymbol\Delta}_{jn}=\frac{\boldsymbol\Delta_{jn}}{\max(\delta_{jn},10^{-6})}.
\end{aligned}
\tag{20}
$$

$\Delta h,\Delta w$ 为窗口格相对中心的行列偏移，始终按最大窗口半径 $4,8$ 归一化，而非随阶段改变尺度。
径向余量与欧氏净空互补：前者描述射线测距相对于查询点径向位置的远近，后者描述返回点到查询点的三维距离。
方向内积进一步区分障碍相对于局部运动方向的位置。
无返回时位移、净空和障碍方向置零；径向余量仍表示查询点相对于测量量程的位置，不作为已观测障碍距离。

两组特征在 $64$ 维隐空间相加，再映射成 $256$ 维关系消息：

$$
\boldsymbol m_{jn}
=W_m\operatorname{SiLU}(W_{\xi}\boldsymbol\xi_{jn}+W_{\zeta}\boldsymbol\zeta_{jn}),
\quad W_{\xi}:8\rightarrow64,\quad W_{\zeta}:4\rightarrow64,\quad W_m:64\rightarrow256.
\tag{21}
$$

这些映射均不含偏置，方向投影 $W_{\zeta}$ 初始化为零并参与后续学习。
该设计将显式几何作为可学习消息提供给网络，没有将净空或方向手工转换成避障控制量。

#### 4.4.4 联合局部注意力与参数共享

第 $k$ 次调用将重建的联合 Query 加上阶段向量 $\boldsymbol e_k^{\mathrm{stage}}$，经 LayerNorm 后生成 attention Query。
K/V 来自式（19），不额外施加全局模块中的独立 K/V LayerNorm。
对注意力头 $a=1,\ldots,8$，令 $\boldsymbol q_a,\boldsymbol k_{jn,a},\boldsymbol v_{jn,a}\in\mathbb R^{32}$
分别为相应线性投影，则

$$
\begin{aligned}
s_{jn,a}&=\frac{\boldsymbol q_a^{\mathsf T}\boldsymbol k_{jn,a}}{\sqrt{32}}
 +\boldsymbol w_a^{\mathsf T}\boldsymbol m_{jn},\\
\alpha_{jn,a}&=\frac{\exp(s_{jn,a})}
{\sum_{(j',n')\in\mathcal S_k}\exp(s_{j'n',a})},\\
\boldsymbol o_a&=\sum_{(j,n)\in\mathcal S_k}\alpha_{jn,a}
\left(\boldsymbol v_{jn,a}+\boldsymbol m_{jn,a}\right).
\end{aligned}
\tag{22}
$$

$\mathcal S_k$ 是全部有效位置—窗口格对的集合，$\boldsymbol m_{jn,a}$ 是关系消息按头拆分后的 $32$ 维向量。
每个头仅在整个 $\mathcal S_k$ 上执行一次 softmax，不对十个位置分别归一化。
关系消息同时影响“关注哪里”的分数与“读取什么”的 value，使几何关系不局限于一个标量注意力偏置。
联合 softmax 允许在不同未来位置间分配注意力，但不保证每个位置获得相同权重或一定得到充分关注。

各头输出拼接后经 $256\rightarrow256$ 投影，与含阶段向量的 Query 残差相加，再经过
$256\rightarrow1024\rightarrow256$ 的 pre-normalized FFN 和 $256\rightarrow256\rightarrow18$ 联合输出头。
注意力投影、关系编码、FFN、角色向量和输出头均在三次调用间共享；三个阶段向量组成一个 $3\times256$ 参数表。
整个共享 Refiner 包含 $896\,448$ 个参数，不因调用三次而复制三套权重。

各阶段使用

$$
\boldsymbol z_k^{\mathrm{sg}}=\operatorname{stopgrad}(\boldsymbol z_k),
\qquad
\Delta\boldsymbol z_k=\mathcal R_{\theta}
\left(\boldsymbol q(\boldsymbol z_k^{\mathrm{sg}},\boldsymbol s),
\mathcal H(\boldsymbol z_k^{\mathrm{sg}},\boldsymbol s),E_f,k\right),
\qquad k=0,1,2,
\tag{23}
$$

其中 $\mathcal H$ 为 Hermite 查询构造，$\mathcal R_{\theta}$ 为同一个修正模块。
参数 Query 与查询位置是当前 action 的两个并行分支，不是从 Query 解出位置。
每次更新后舍弃局部 Query 内容，再由新参数重建；没有跨调用传递隐藏内容向量。
stop-gradient 阻止后层损失沿 action 链反向改变前层输出，但共享感知、状态、参数编码器与 Refiner 仍接收各阶段损失的梯度。
这种梯度组织不等同于冻结前端，也不取消中间阶段监督。

### 4.5 有界参数更新与最终解码

#### 4.5.1 统一的残差参数化

全局提议与局部修正使用同一形式的参数更新。
令 $\boldsymbol y\in\mathbb R^{12}$ 表示归一化的内部位置、终端位置和终端速度，$\Delta\boldsymbol y$ 为对应修正量，则

$$
\boldsymbol y^+=\tanh\!\left(\operatorname{atanh}(\boldsymbol y)+\Delta\boldsymbol y\right).
\tag{24}
$$

运算逐分量进行，实际计算在 $\operatorname{atanh}$ 前使用 $1-10^{-5}$ 的边界裁剪。
在裁剪边界以内，该映射的零修正保留原参数，非零修正后的分量仍位于归一化区间内。
其边界定义在内部坐标系，不意味着所有位置或速度向量都满足同一个欧氏范数上限。

终端加速度采用径向映射，以保持三维向量范数有界。
定义

$$
\mathcal T(\boldsymbol x)=\tanh(\|\boldsymbol x\|_2)\frac{\boldsymbol x}{\|\boldsymbol x\|_2},
\qquad
\mathcal T^{-1}(\boldsymbol x)=\operatorname{atanh}(\|\boldsymbol x\|_2)\frac{\boldsymbol x}{\|\boldsymbol x\|_2},
\qquad
\bar{\boldsymbol a}_e^+=\mathcal T\!\left(\mathcal T^{-1}(\bar{\boldsymbol a}_e)+\Delta\boldsymbol a_e\right).
\tag{25}
$$

$\mathcal T^{-1}$ 的定义域为单位开球；零向量处按连续极限定义，并在实现中进行数值保护。
这样限制的是终端加速度范数，不是整条曲线的加速度或 jerk 峰值。

对时间 logits，令 $\Pi(\boldsymbol x)=\boldsymbol x-\frac13\boldsymbol1\boldsymbol1^{\mathsf T}\boldsymbol x$，使用

$$
\boldsymbol\ell^+=\Pi\!\left(\boldsymbol\ell+0.5\tanh(\Pi(\Delta\boldsymbol\ell))\right),
\qquad \boldsymbol T^+=H\operatorname{softmax}(\boldsymbol\ell^+).
\tag{26}
$$

共同 logit 偏移不改变 softmax，去均值只消除这一冗余自由度，不固定时间比例。
有界 logit 修正控制单次更新幅度，但不等同于预设的最小段时长约束，也不等价于限制物理轨迹变化的信赖域。
因此，本文的 Refiner 是学习式残差修正，不是具有下降或收敛保证的优化迭代。

全局输出头末层以标准差 $10^{-3}$ 初始化；共享 Refiner 输出头末层初始化为零。
忽略边界处的数值裁剪时，初始化的局部修正保持全局输出，随后由阶段监督学习修正量。
这些初始化与参数化是明确的实现选择，不构成对最终训练行为的理论保证。

#### 4.5.2 训练与推理中的解析边界

式（24）—（26）首先将参考参数更新为 $\boldsymbol z_0$，再将式（23）得到的修正量逐次施加于
$\boldsymbol z_0^{\mathrm{sg}},\boldsymbol z_1^{\mathrm{sg}},\boldsymbol z_2^{\mathrm{sg}}$，得到最终参数 $\boldsymbol z_3$。
将五个三维几何／边界向量从 $\mathcal C$ 旋回 $\mathcal B$，恢复物理尺度，并以未经网络归一化的
$\boldsymbol v_0^{\mathcal B},\boldsymbol a_0^{\mathcal B}$ 构造 MINCO 起始边界。
给定式（6）的系数后，可解析计算任意时刻的位置、速度、加速度和 jerk；世界系轨迹为

$$
\boldsymbol p^{\mathcal W}(t)
=\boldsymbol x_0^{\mathcal W}+R_{\mathcal W\mathcal B}\boldsymbol p^{\mathcal B}(t).
\tag{27}
$$

训练时，网络返回 $\boldsymbol z_0,\boldsymbol z_1,\boldsymbol z_2,\boldsymbol z_3$，
在外部沿 batch 维打包为 $4B$ 个问题，一次批量调用 MINCO 求解器。
四阶段均在 $0.1,0.2,\ldots,2.0\,\mathrm s$ 的二十个未来时刻接受轨迹监督；
Hermite 的十个查询位置不替代这些监督点。
推理时执行相同的三次参数修正与局部查询，仅对最终 $\boldsymbol z_3$ 解码，不求解前三条 MINCO 轨迹。
这一差异只改变监督分支是否计算，不改变网络预测规则，也不以 Hermite 曲线作为执行输出。
训练目标及 jerk 正则的完整定义在后续训练方法章节给出。

### 4.6 计算组成与方法边界

当前网络共有 $4\,560\,448$ 个可训练参数。
编码器在一次规划中仅执行一次，粗尺度 token 数为原生精细尺度的四分之一。
对一个 Query 和 $N$ 个环境 token，注意力分数与加权聚合的计算量随 $Nd$ 线性增长，
而 K/V 投影包含 $Nd^2$ 量级的操作；这里不存在环境 token 全局 self-attention 的 $N^2$ 分数矩阵。
三层全局模块各自处理 $1350$ 个 token，三次局部修正合计处理至多 $1530+450+150=2130$ 个位置—射线对。
该计数包含重叠窗口中的重复射线，不是独立障碍物数量。

参数共享减少的是重复存储的模块参数，不会免除三次前向计算。
粗尺度交互、局部窗口和取消中间 MINCO 解码分别减少不同部分的工作量，也不能将 token 数比例直接解释为整网加速比。
实际延迟还受投影、特征索引、关系编码、内核调度和最终 float64 求解影响，应分别测量各模块及完整推理链路。

这一架构保留了三类清晰接口：观测进入环境记忆，当前轨迹参数决定查询与修正，最终物理参数决定连续曲线。
单 Query 和固定三次修正提供紧凑、确定的计算流程，但不显式表示多模态决策不确定性；
局部查询依赖代理几何与传感器视场，单帧输入也不提供时间一致性的直接约束。
因此，方法有效性需要分别由轨迹质量、逐层风险变化、推理效率及闭环行为验证，不能仅由结构设计推导。

## 参考文献

- Wang, Z., Zhou, X., Xu, C., and Gao, F. **Geometrically Constrained Trajectory Optimization for Multicopters.**
  *IEEE Transactions on Robotics*, 2022. [论文](https://arxiv.org/abs/2103.00190)
- Lu, J., Zhang, X., Shen, H., Xu, L., and Tian, B. **You Only Plan Once: A Learning-Based One-Stage Planner With Guidance Learning.**
  *IEEE Robotics and Automation Letters*, 2024. [作者提供的论文](https://wayhome316.github.io/projects/onlyplanonce/youonlyplanonce.pdf)
- Chen, Y., Li, J., Qin, W., Hua, Y., Dong, X., and Li, Q. **Learning to Initialize Trajectory Optimization for Vision-Based Autonomous Flight in Unknown Environments.**
  *IEEE/RSJ International Conference on Intelligent Robots and Systems*, 2025. [论文](https://arxiv.org/abs/2309.10683)
- Liu, Z., Xu, G., Wang, Z., Hou, J., Xu, C., and Gao, F. **Neural-Primitive: An Efficient End-to-end Local Planner with Primitive-based Imitation Learning for Autonomous Flight.**
  *arXiv preprint arXiv:2608.20948*, 2026. [预印本](https://arxiv.org/abs/2608.20948)
- Wu, Y., Sun, X., Spasojevic, I., and Kumar, V. **Deep Learning for Optimization of Trajectories for Quadrotors.**
  *IEEE Robotics and Automation Letters*, 2024. [论文](https://arxiv.org/abs/2309.15191)
- Han, Z., Xu, L., Pei, L., and Gao, F. **Dynamically Feasible Trajectory Generation With Optimization-Embedded Networks for Autonomous Flight.**
  *IEEE Robotics and Automation Letters*, 2025. [作者项目与方法说明](https://zju-fast-lab.github.io/e2e_opt/)
- Hu, Y., et al. **Planning-Oriented Autonomous Driving.**
  *IEEE/CVF Conference on Computer Vision and Pattern Recognition*, 2023.
  [论文](https://openaccess.thecvf.com/content/CVPR2023/html/Hu_Planning-Oriented_Autonomous_Driving_CVPR_2023_paper.html)
- Shi, S., Jiang, L., Dai, D., and Schiele, B. **Motion Transformer with Global Intention Localization and Local Movement Refinement.**
  *Advances in Neural Information Processing Systems*, 2022. [论文](https://arxiv.org/abs/2209.13508)
- Zhou, Y., Shao, H., Wang, L., Waslander, S. L., Li, H., and Liu, Y. **SmartRefine: A Scenario-Adaptive Refinement Framework for Efficient Motion Prediction.**
  *IEEE/CVF Conference on Computer Vision and Pattern Recognition*, 2024. [论文](https://arxiv.org/abs/2403.11492)

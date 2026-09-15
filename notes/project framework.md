# 报告框架书

------

## 0. 全局共享设定（所有人写自己的部分前必须先统一）

这些设定不属于某一节，而是被 2、4、5 章反复引用的"公共参数"，谁改了都要在群里说一声，否则会出现"我的图和你的图对不上"的问题。

- **积分区间**：t ∈ [0, 40]。

- **输出网格**：t = 0，再加上 **至少 200 个从 1e-8 到 40 的对数间隔点**。所有方法、所有对比图必须用**同一套输出时刻**，否则误差对比没有意义。

- **Reference solution 构造方式**：`scipy.integrate.solve_ivp`，方法用 Radau/BDF/LSODA，先用很紧的 rtol/atol 跑一遍，再把容差整体收紧 100 倍重跑一遍，**只保留两次结果中没有变化的那几位数字**，作为"可信参考解"。这是第 5.1 节的活，但产出的 reference trajectory 会被 4.2、5.2、5.3、5.4、6.1 反复调用，谁先做完请尽快共享数据文件。

- Check values（仅用于自查，不能当成"题目给的标准答案"写进正文引用来源）

  ：

  - |λ|_max ≈ 3.39×10³ （t=40 附近）
  - stiffness ratio ≈ 3.0×10³ @ t=1e-4，≈5.4×10³ @ t=1e-2，≈1.58×10⁵ @ t=40
  - y(40) ≈ (0.7158271, 9.1855×10⁻⁶, 0.2841637)（只报告你自己收紧容差后仍不变的位数，不要抄满7位）

- **建模陷阱提醒**：ODE 里 `3e7 * y2^2` 这一项**不要**因为反应式写的是 B+B 就多乘一个 2，题目原文明确说了这是mnemonic、ODE以给定公式为准。

- Threefold validation（贯穿全文的证据链，写每一节时想一下"我这节属于哪一条"）

  ：

  1. 独立 oracle（scipy 紧容差 Radau/BDF）
  2. 已知特例 / 守恒量（线性不变量 y1+y2+y3=1）
  3. 不同阶方法在细网格极限下互相吻合（Euler vs RK4 vs Implicit Euler vs Trapezoidal） 第 6.3 节要把这三条串成一段总结，所以前面每节产出证据时，最好顺手标一下"这属于第几条"，6.3 写起来会省很多事。

------

## 1 Introduction

### 1.1 Background and Motivation

**任务**：介绍 Robertson 问题的来历（H. H. Robertson, 1966，刚性ODE求解器的经典benchmark），说明三个反应速率常数跨越约 10⁸ 量级（0.04 vs 3×10⁷），点出"这会让显式方法灾难性失败"作为悬念，不展开细节（细节留给后面章节）。 **与全文关系**：给读者一个"为什么这道题值得做"的理由，是整篇报告的钩子。 **需要的图**：不需要。如果想加分可以放一张三条反应速率的示意图（时间尺度分离的直观漫画），非必需。 **完成标准 (DoD)**：

- [ ] 讲清楚这是哪类问题（刚性ODE）、为什么经典
- [ ] 点出三个速率常数的数量级差异，但不下结论（结论留给第4、6章）
- [ ] 篇幅控制在半页以内，不要提前剧透第4、6章的分析结果

### 1.2 Problem Statement and Report Roadmap

**任务**：把"我们要回答的数学问题"用一两句话精确表述（例如："能否构造一个可信的数值解，并定量解释为什么某些方法失败、某些方法成功"），然后一句话带过每章逻辑（2建模→3方法→4理论预测→5实验验证→6解释→7结论）。 **与全文关系**：这是全文的"地图"，后面每章开头如果能呼应这里提的问题，读起来会更连贯。 **需要的图**：不需要。 **完成标准 (DoD)**：

- [ ] 问题表述精确到"数值上可验证"的程度，不是泛泛而谈
- [ ] Roadmap 覆盖全部7章，每章一句话

------

## 2 Mathematical Formulation

### 2.1 Reaction Kinetics Model

**任务**：写出模型方程：

```
y1' = -0.04 y1 + 1e4 y2 y3
y2' =  0.04 y1 - 1e4 y2 y3 - 3e7 y2^2
y3' =  3e7 y2^2
y(0) = (1, 0, 0)
```

说明三个反应的物理意义（A→B, B+B→B+C, B+C→A+C），并**明确写一句建模陷阱提醒**（3e7项不要多乘2，理由见0节）。 **与全文关系**：全文所有计算的起点，后面每一章的公式都从这里的 f(y) 出发。 **需要的图**：不需要方程本身的图，如果想让读者更直观，可以画一个反应网络示意图（A/B/C 三个节点 + 箭头），非必需。 **完成标准 (DoD)**：

- [ ] 方程与初值写对，和题目原文逐字核对过
- [ ] 建模陷阱提醒写清楚（这是容易被扣分的细节）

### 2.2 Conservation Invariant (y1+y2+y3=1) and Structural Zero Eigenvalue

**任务**：证明两件事，建议写成正式 Proposition + Proof：

1. **代数恒等式**：对任意 y（不只是轨迹上的点），1ᵀf(y)=0（三对项相消，直接展开验证）。
2. **推论**：沿精确轨迹积分得到 y1+y2+y3≡1；对恒等式两边求梯度得到 1ᵀJ(y)≡0，即完整 3×3 Jacobian **恒有一个零特征值**，与走到哪个 y 无关。 再补一句承上启下的话：这就是为什么直接对完整 Jacobian 算 stiffness ratio 会遇到"分母是0"的问题——这个坑由 2.3 来解决。 **与全文关系**：这是全文数学部分的地基。2.3 的降维操作、3.x 的线性不变量保持机制、4.2 的 stiffness ratio 定义，全部依赖这里证明的结果。 **需要的图**：不需要。 **完成标准 (DoD)**：

- [ ] 恒等式和推论都写成完整证明，不是"显然成立"一笔带过
- [ ] 明确点出"零特征值与轨迹无关，是结构性的"这一点
- [ ] 结尾有一句话过渡到 2.3（"这就是为什么需要约化"）

### 2.3 Reduced Jacobian and Stiffness Ratio Definition

**任务**：

1. 消去 y3=1-y1-y2，写出2维自洽系统，求 2×2 reduced Jacobian J_r 的四个分量表达式。
2. **给出严格性论证**（不是"直觉上应该没错"）：因为 2.2 证明了 1ᵀJ(y)≡0，所以 J(y) 把整个 R³ 映到 T={v:1ᵀv=0} 这个2维子空间里；配合 y(t) 落在仿射流形 M={y:1ᵀy=1} 上，可以论证用 (y1,y2) 参数化 M 求出的 J_r，就是完整 Jacobian 限制在不变流形 M 上的精确线性化，不是近似消元。
3. 定义 stiffness ratio：S(t) = max|Reλ_j| / min|Reλ_j|（J_r 的两个非零特征值），并说明要设定一个数值阈值来判定"零特征值"（用于处理 t→0 处的退化），从第一个正的对数时间点开始画。 **与全文关系**：这里定义的 J_r 会被 3.3/3.4 的 Newton 求解器直接复用（Dg(Y)=I-hJ_r(Y)），也是 4.2 的直接计算对象。这是数学部分和数值部分的"接口"。 **需要的图**：不需要（图放在4.2里画）。 **完成标准 (DoD)**：

- [ ] J_r 四个分量公式写对，最好用符号计算工具（sympy）核对一遍
- [ ] "约化操作合法性"的论证写清楚，不是一句"显然可以消元"带过
- [ ] Stiffness ratio 的定义、零特征值判定阈值、t→0 处理方式都明确写出

------

## 3 Numerical Integration Methods

> **给3.1-3.4负责人的共同要求**：每个方法实现完后，先用一个有解析解的简单测试方程（比如 y'=-y，或者别的非刚性/温和刚性的例子）验证一下你的实现是对的，再拿去跑 Robertson。这是 threefold validation 里"已知特例"这条的一部分，做了记得在小节里提一句"已用xx测试方程验证实现正确性"。

### 3.1 Explicit Euler Method

**任务**：写出格式 y_{n+1}=y_n+h f(y_n)，说明 order 1（一句话带过局部截断误差量级 O(h²)/全局 O(h)）。 **与全文关系**：这是"失败案例"的主角，戏份主要在第4、5章。这里只需要把方法本身讲清楚，结尾留一句 forward pointer："其稳定性区域和对本系统的实际步长限制将在第4章定量分析"——不要在这里剧透结论。 **需要的图**：不需要。 **完成标准 (DoD)**：

- [ ] 格式、阶数写对
- [ ] 结尾有 forward pointer 到第4章，不提前下结论

### 3.2 Explicit Fourth-Order Runge-Kutta Method (RK4)

**任务**：写出经典 RK4 的 Butcher tableau 和格式（k1,k2,k3,k4 + 组合公式）。**建议做到⋆⋆⋆深度**：手推 order≤3 的 order condition（Taylor展开比较系数，至少推 Σb_i=1, Σb_ic_i=1/2, Σb_ic_i²=1/3 这几条），第4阶的8个条件可以列出但引用经典结果说明满足，不必逐条验证（太长，放正文会超页数）。 **与全文关系**：是"精度高但不管刚性"的对照组，第4、5章会展示它在刚性区间掉阶或失效。 **需要的图**：不需要（收敛阶验证图放5.2）。 **完成标准 (DoD)**：

- [ ] Butcher tableau 写对
- [ ] 至少手推出 order 3 的条件验证，第4阶条件列出并注明引用来源
- [ ] 同3.1，结尾留 forward pointer

### 3.3 Implicit Euler Method with Newton-Raphson Iteration

**任务**（这节内容最重，建议单人专注写）：

1. 写出隐式方程 g(Y)=Y-y_n-h f(Y)=0，以及 Newton 迭代式 Y^(k+1)=Y^(k)-[Dg(Y^(k))]^(-1) g(Y^(k))，其中 Dg(Y)=I-h·J(Y)，**明确指出这里的 J 就是2.3里定义的那个 Jacobian**——这是数学部分和方法实现的"接口"，一定要点出来。
2. 写清楚停止判据（‖g(Y^(k))‖ < tol）和初始猜测的选取（通常 Y^(0)=y_n）。
3. **证明线性不变量保持机制**（可作为本节或独立Proposition）：定义残差 r=g(Ŷ)=Ŷ-y_n-h f(Ŷ)，推出 1ᵀŶ = 1ᵀy_n + 1ᵀr——**守恒亏损精确等于Newton残差**。这个结论要在5.3.1、5.4被反复引用，务必写严格。
4. （可选⋆⋆⋆）一句话提 Newton 局部二次收敛定理，并补一句"本问题专属注记"：J 的谱包含 3×10⁷ 量级特征值时 Dg=I-hJ 条件数可能很大，收敛域可能变窄，这点会在5.4的容差扫描里验证。 **与全文关系**：这是"成功案例"的主角，3.3/3.4 的 stability 性质（4.1）、守恒机制（这里证明，5.3.1验证）、Newton容差敏感度（5.4）都从这节的推导延伸出去。 **需要的图**：不需要。 **完成标准 (DoD)**：

- [ ] Newton 迭代公式和停止判据写清楚，且明确点出 Dg 里的 J 和2.3是同一个对象
- [ ] 守恒亏损=Newton残差 这条**必须写成完整推导**，不能一笔带过（后面两节要用）
- [ ] （若做⋆⋆⋆）Newton收敛性注记写出

### 3.4 Trapezoidal / Crank-Nicolson as a Second Implicit Method

**任务**：写出格式 y_{n+1}=y_n+h/2[f(y_n)+f(y_{n+1})]，Newton迭代同3.3的思路但 Dg(Y)=I-(h/2)J(Y)。说明这是 order 2（比implicit Euler的order 1高），这是为5.2"不同阶方法自洽性"检验准备的素材，写的时候点一句这个用途。 **与全文关系**：和 implicit Euler 一起构成"两个隐式方法互相印证"的自洽性检验对象（threefold validation第3条）；同时它在4.1会被证明不是L-stable（只是A-stable），这个细节要留到4.1详细展开，这里不用讲。 **需要的图**：不需要。 **完成标准 (DoD)**：

- [ ] 格式、阶数写对，点出和3.3构成"不同阶方法自洽性检验"的关系
- [ ] 不要在这里提前讲A-stable/L-stable的区别（留给4.1）

### 3.5 Adaptive Step-Size Control

**任务**：选一种方案写清楚（step-doubling + 局部误差估计，或者embedded pair），写出误差估计公式和步长调整规则（比如 h_new = h·(tol/err)^(1/(p+1)) 这类）。 **与全文关系**：这是方法工具箱的最后一块，5.x的实验里如果用adaptive方法去构造reference或做效率对比，会调用这里的算法。 **需要的图**：**需要**——步长 h(t) 随 t 变化的图（建议 t 用 log 轴，因为初始过渡层极短），展示步长在刚性区间自动变小、缓和区间自动变大；如果篇幅够，再加一张"局部误差估计值 vs 设定tolerance"的图，证明误差确实落在tolerance附近（这是brief原文明确要求的：the step size should actually change, and the error should land near the tolerance you set）。 **完成标准 (DoD)**：

- [ ] 方法选择说明清楚，公式完整
- [ ] h(t) 图做出来，能看出步长自适应变化
- [ ] 误差落在tolerance附近的证据（图或表）

------

## 4 Linear Stability Analysis

### 4.1 Absolute Stability Regions of the Three (or Four) Methods

**任务**：对标量测试方程 y'=λy，z=hλ，推导四个方法的增长因子 R(z)：

- Explicit Euler: R(z)=1+z，圆盘稳定域
- RK4: R(z)=1+z+z²/2+z³/6+z⁴/24
- Implicit Euler: R(z)=1/(1-z)，**A-stable 且 L-stable**（z→-∞时R→0）
- Trapezoidal: R(z)=(1+z/2)/(1-z/2)，**A-stable 但不是L-stable**（z→-∞时R→-1，可能有轻微振荡） 四个方法的稳定域画在同一张图（或2×2子图）里方便对比。（可选⋆⋆⋆）加一句B-stability的理论边界说明：这些都是线性测试方程的结果，真正的非线性稳定性理论是Dahlquist的B-stability，需要单调性条件，本问题未必满足，所以后面要靠经验的step sweep补充——这句话是为4.3、6.3的验证策略铺垫。 **与全文关系**：这是"为什么隐式方法成功"的理论核心，直接喂给4.2/4.3做出定量预测，也是6.2的理论依据。 **需要的图**：**需要**——复平面 (Re(z), Im(z)) 上四个方法的稳定域，建议画在一起（不同颜色/图例）或2×2排列，标出A-stable/L-stable的区别。 **完成标准 (DoD)**：
- [ ] 四个R(z)推导正确
- [ ] 明确区分A-stable和L-stable，并指出Trapezoidal不是L-stable这个后面要验证的预测点
- [ ] 稳定域图清晰、有图例

### 4.2 Eigenvalue Analysis of the Robertson System

**任务**：

1. 沿0节的reference trajectory，逐点计算2.3的 J_r 的两个特征值，画出 |λ1(t)|, |λ2(t)| 和 stiffness ratio S(t) 随（log）t 变化的图。
2. **（重要，题目原文点名要求）非正规性检验**：算一下 J_r 是否正规（一般不是，因为不对称），可以算特征向量矩阵的条件数 κ(V) 作为廉价指标。写一句结论："κ(V)较大说明特征值分析可能不完全可靠（非正规矩阵存在瞬态增长风险），需要4.3的经验step sweep来补充验证"——这直接呼应Problem Pack原文那句"eigenvalues alone can be misleading for non-normal systems"。 **与全文关系**：这是4.1的稳定域理论和具体问题的"连接点"，4.3的步长上界估计直接用这里算出的|λ|_max。 **需要的图**：**需要**——(a) 两个非零特征值模长 vs log(t)；(b) stiffness ratio S(t) vs log(t)。x轴从第一个正的对数时间点（如1e-8或更晚）到40，y轴用log scale。可以叠成两个子图或一张双y轴图。 **完成标准 (DoD)**：

- [ ] 特征值/ratio图做出来，数值级别和check values同量级（不要求一致，量级对上即可）
- [ ] 非正规性检验做了并写出结论

### 4.3 Step-Size Constraints: Theory vs Experiment

**任务**：

1. 理论侧：用4.2算出的|λ|_max给出显式Euler稳定步长上界估计 h < 2/|λ|_max，**明确声明这只是frozen-Jacobian局部线性化估计，不是充分的非线性稳定性证明**。
2. 实验侧：做一个真实的step-size sweep（显式Euler在一系列h下实际运行，观察何时发散/出现负值/误差爆炸），和理论估计对比，report discrepancy（如果有）。 **与全文关系**：这一节是"理论预测→实验验证"这条叙事主线的关键节点，5.3的诊断实验结果要回过头来和这里的预测对照，6.2写"为什么显式失败"时要引用这里的结论。 **需要的图**：**需要**——h（step size）为x轴（log scale），y轴可以是"是否稳定/是否非负/误差大小"的某种指标（比如最终误差、或者一个二值稳定性标记），把理论估计的h阈值用竖线标出，和实测阈值对比。 **完成标准 (DoD)**：

- [ ] 理论估计写出并声明其局限性
- [ ] 实测sweep做出来，和理论值对比，有一句话说明"吻合"或"有偏差，原因是XXX"

------

## 5 Numerical Experiments

### 5.1 Reference Solution Construction

**任务**：按0节的设定，实际构造reference solution，包括tolerance-refinement test（收紧100倍看哪些位数不变）和BDF独立交叉验证。最后汇总出"我们采信的reference trajectory"，报告到自查的check value时**只写自己验证过没变化的位数**。 **与全文关系**：这是全篇所有误差计算、所有对比图的"标准答案"来源，必须最先完成，其他人（4.2、5.2、5.3、5.4、6.1画图）都依赖这里的数据文件。**建议这节尽早完成并共享数据**。 **需要的图**：建议做一张"master figure"——reference解 y1(t), y2(t)（可能需要放大或对数y轴，因为量级只有1e-5~1e-6）, y3(t) 随log(t)变化的图，这张图会被后面很多小节复用或参照。 **完成标准 (DoD)**：

- [ ] tolerance-refinement test 做了，写出保留了几位数字
- [ ] BDF交叉验证做了，两者吻合
- [ ] Reference数据文件产出并共享给全组
- [ ] Master figure 做出来

### 5.2 Convergence Order Verification

**任务**：对至少两种方法（建议四种都做，工作量不大）做 error-vs-h 的收敛阶验证，error 用相对5.1的reference算。 **与全文关系**：这是threefold validation第3条"不同阶方法自洽性"的直接落地，也是brief明确要求的deliverable。 **需要的图**：**需要**——log-log 的 error vs h 图，每个方法一条线，旁边画理论斜率参考线（1, 2, 4阶对应的斜率），观察实测斜率是否吻合（注意：RK4在刚性区间可能掉阶，如果观察到这个现象要在图注/正文里点出来，这是个值得写的发现）。 **完成标准 (DoD)**：

- [ ] 至少两个方法（建议全部四个）做出收敛阶图
- [ ] 实测斜率和理论对比，有discrepancy要解释

### 5.3 Stability and Physical Validity Tests

> 三个子节共用同一批运行结果（各方法在几个不同h下的解），建议一次性生成数据，分三张图/表展示不同诊断量。**务必在文字里强调"conservation alone is weak"**——精确守恒不代表解准确或非负，三个诊断要一起看。

#### 5.3.1 conservation

**任务**：计算 |y1+y2+y3-1| 随t变化，对每个方法。**要点出显隐式方法的不对称性**（呼应3.3证明的结论）：显式方法（Euler/RK4）的守恒偏差只受浮点舍入限制，理论上极小；隐式方法的偏差由Newton残差控制，如果Newton容差设置不够严，会出现可测的偏差。 **需要的图**：**需要**——|y1+y2+y3-1| vs t（y轴log scale）,每个方法一条线。 **DoD**：

- [ ] 图做出来，显隐方法的差异和3.3的理论预测吻合（或者不吻合要解释为什么）

#### 5.3.2 non-negativity

**任务**：检查各方法在不同h下是否出现分量为负的情况，尤其是显式Euler在step-size较大时。 **需要的图**：可以是图（min(y_i(t)) vs t，标出0线）或表格（不同h下"是否出现负值"的二值表），表格更简洁，视数据形态选。 **DoD**：

- [ ] 至少展示显式Euler在"理论不稳定步长"附近确实出现负值，隐式方法不出现

#### 5.3.3 full-state error

**任务**：算 ‖y_method(t) - y_ref(t)‖ vs t，各方法在"匹配计算成本"（比如相近步数）下对比。 **需要的图**：**需要**——误差 vs t（y轴log scale），各方法一条线。 **DoD**：

- [ ] 图做出来，能看出隐式方法整体误差更低/更稳（在刚性区间）

### 5.4 Diagnostic Analysis（Newton容差扫描 + 最终y2值在多个容差下的报告）

**任务**：这节承接3.3提到的Newton容差敏感度分析——做一个**Newton停止容差扫描**：不同容差下，(a) 最终y2值和迭代次数，(b) 与reference对比的整体误差。**这正好覆盖Problem Pack原文明确要求的deliverable**："The final value of y2 reported by your implicit method at several tolerances, cross-checked against your own high-accuracy reference"——一定不要漏掉这条，是题目原文点名的。 **与全文关系**：呼应3.3证明的"Newton容差需要比时间离散误差更严格"的要求，也是"诊断"章节存在的意义——展示中间过程而非只看最终结果。 **需要的图/表**：**需要**——表格：不同Newton容差 → 最终y2值、迭代次数、与reference的误差；如果想做成图，可以是 误差 vs Newton容差 的log-log图，展示当容差足够严后误差趋于平台（被时间离散误差主导）。 **完成标准 (DoD)**：

- [ ] Newton容差扫描做了，最终y2值在几个容差下都报告并和reference对比
- [ ] 能看出/说明"容差足够严之后误差不再下降"这个现象（代数误差被离散误差掩盖）

### 5.5 Efficiency Comparison

**任务**：在相近精度下比较各方法的计算成本（步数、函数求值次数、或wall time）。 **与全文关系**：呼应"隐式方法每步贵但整体可能更省"这个直觉，为6.2提供量化支撑。 **需要的图**：**需要**——经典 work-precision diagram：x轴计算成本（log scale），y轴误差（log scale），每个方法一条曲线（多个h对应曲线上多个点）。 **完成标准 (DoD)**：

- [ ] work-precision图做出来
- [ ] 一句话结论：在什么精度要求下哪个方法更划算

------

## 6 Discussion

### 6.1 Physical Interpretation via Quasi-Steady-State Approximation (QSSA)

**任务**：解释为什么y2先冲一个尖峰再迅速relax到极小准稳态值。基础版本：量级平衡估计（0.04y1 ≈ 3e7 y2²，早期y3≈0），得到y2量级和check value（~1e-5~1e-6）吻合。**⋆⋆⋆加分版本**：用奇异摄动框架（ε=1/k2为小参数，写成slow-fast标准形式），引用Tikhonov定理说明解在O(ε)时间内收敛到慢流形附近——这个框架还能解释4.2里"为什么早期特征值分离剧烈、后期趋于温和"的现象，值得一提这个联系。 **与全文关系**：把前面纯数值的观察（y2的尖峰+relax行为）升级为有理论支撑的物理解释，是全文"从数值到理解"的收尾环节之一。 **需要的图**：建议放一张y2在t∈[0, 1e-3]附近的放大图（局部zoom-in），展示尖峰和快速relax的形状，如果做了QSSA预测曲线可以叠加对比。 **完成标准 (DoD)**：

- [ ] 至少完成量级估计版本的解释
- [ ] （可选）Tikhonov框架版本完成，且和4.2的现象建立联系

### 6.2 Why Explicit Methods Fail and Implicit Methods Succeed

**任务**：把4.1（A/L-stability理论）+ 4.2/4.3（特征值和实测step sweep）的结论串起来，正面回答标题问题。记得提一句4.1里"Trapezoidal不是L-stable"的预测，如果5.3.1/5.3.3观察到对应的轻微振荡现象，在这里点出来呼应；如果没观察到，也要说明可能原因（比如步长不够小到能体现这个效应）。 **与全文关系**：是4、5两章的"综合"，不引入新数据，只做解释和串联。 **需要的图**：不需要新图，引用前面的图即可。 **完成标准 (DoD)**：

- [ ] 明确串联4.1/4.2/4.3/5.3的结论，形成完整因果链
- [ ] 对Trapezoidal的L-stability预测有呼应（证实或解释未观察到的原因）

### 6.3 Three-Fold Validation Strategy Recap

**任务**：把全文出现过的三条证据链汇总成一段："独立oracle"对应5.1（reference construction）,"已知特例/守恒量"对应5.3.1（conservation）+2.2的理论,"不同阶方法自洽"对应5.2（convergence order）。明确写一句："这就是我们相信最终数字的理由链"。 **与全文关系**：这是全文方法论最重要的一段总结，直接对应grading rubric里强调的"评分不是数字本身而是证据链"。 **需要的图**：不需要。 **完成标准 (DoD)**：

- [ ] 三条证据链都点名对应到具体的前文小节（不是空泛地说"我们做了验证"）

### 6.4 Limitations

**任务**：诚实列出局限：frozen-Jacobian局部分析的边界（4.1/4.3）、Newton收敛半径的限制（3.3）、adaptive控制的简化程度（3.5）、非正规性未完全展开分析（4.2，如果只做了κ(V)简单检验没做完整pseudospectra）。 **需要的图**：不需要。 **完成标准 (DoD)**：

- [ ] 至少列出3-4条具体的（不是套话式的）局限

------

## 7 Conclusions

**任务**：简短回收1.2提出的问题，给出结论性数字（引用5.1.6/5.1的reference值），给出方法选择建议。 **需要的图**：不需要。 **完成标准 (DoD)**：

- [ ] 呼应1.2的问题陈述
- [ ] 篇幅控制在半页到一页

------

## References

**任务**：Robertson原始文献(1966)、scipy/Radau/BDF相关文档或论文、RK方法/stability理论的教材引用（如Hairer & Wanner的Solving ODEs II，讲stiff问题的经典教材，如果用了他们的order condition或B-stability定义要引用）、Tikhonov奇异摄动定理如果用了也要引用来源。

## Appendix A: AI Transparency Log

**任务**：记录每个人在写代码/报告过程中用AI做了什么、改了什么、为什么改。格式建议：时间/用途/AI输出摘要/你验证或修改的内容。**这是brief明确要求的deliverable**（"records what you changed in AI output and why — that log is your defence, not your confession"），建议每人边做边记，不要最后一天补。

## Appendix B: Individual Contribution Statement

**任务**：每人写自己负责的部分、贡献占比。占总分25%权重，别糊弄。

## tips:

**（可选，非必需）考虑加一个简短的 Appendix C：Code-to-Figure Mapping**，一个小表格，列出"报告里第几张图 ↔ 对应哪个脚本/哪个函数"。Brief的grading要求"Code: 别人要能用run_all.py复现所有图"，这个小表格几乎不花时间做，但能显著降低"评审跑不出你的图"的风险，对"Implementation: Robust and Reproducible"这15%权重是直接加分。不是必须，但性价比很高，建议加。

其余部分（页码分配、AI Log和ICS作为独立附录）已经覆盖了brief的全部deliverable要求，不需要再改。
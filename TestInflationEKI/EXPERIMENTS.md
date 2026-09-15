# 非线性算法比较

## 保留的方法

- `Inflation EAKI`：inflated square-root EAKI baseline。
- `Sequential projected dropout EAKI`：原始方法；先做 EAKI mean update，再加 projected dropout correction。
- `Joint projected EAKI`：最终保留的新方法，即此前的 Raw Joint；没有 safeguard 或回溯。
- `DEKI`：论文 *Dropout Ensemble Kalman Inversion for High Dimensional Inverse Problems*
  Algorithm 2.1 的本地重实现，使用 covariance-normalized step sizes。
- `CMA-ES`：仓库原始 `Inversion/CMAES.jl`，通过 `PyCall` 调用 Python `cma`；该文件没有修改。

Joint 使用原 anomaly block 和 projected complementary dropout block：

```text
Z_aug = [Z_b, sqrt(w) Z_perp]
Y_aug = [Y_b, sqrt(w) Y_perp]
```

然后用增广 covariance 一次性计算 mean proposal。当前 `w=1`，proposal 直接接受；covariance
仍由 deterministic inflation EAKI update 给出。

## 四个测试问题及初始化

所有问题都写成 nonlinear least-squares 形式

```text
Phi(theta) = 0.5 * ||F(theta)||^2,
```

观测取 `y=0`，观测误差 covariance 取单位矩阵。下面给出的公式已经包含 residual scaling，
因此都是代码实际计算的 `Phi`，而不是只差一个常数因子的简写。

### 1. Monotone cubic

令 `B` 为通过 Gaussian random matrix 的 QR 分解生成的正交矩阵，逐分量定义
`psi(z)=z+2z^3`。目标函数为

```text
Phi(theta) = 0.5 * ||psi(B*theta) - psi(B*theta_star)||^2.
```

`theta_star ~ N(0,I_d)`，是唯一的全局极小点，最优值为 0。每个 ensemble member 独立初始化为

```text
theta_0^(j) ~ N(0,I_d),    j=1,...,J.
```

`B` 和 `theta_star` 都由该 trial 的 problem seed 生成，所以不同 trial 对应不同的正交方向和
不同的最优点。

### 2. Paired Rosenbrock

要求 `d` 为偶数，将变量分成互不重叠的 pair
`(x_i,y_i)=(theta_(2i-1),theta_(2i))`：

```text
Phi(theta) = sum_i [(1-x_i)^2 + 100*(y_i-x_i^2)^2].
```

全局极小点是 `theta_star=(1,...,1)`，最优值为 0。每个 member 从重复的经典 Rosenbrock
起点附近初始化：

```text
theta_0^(j) = repeat((-1.2,1.0), d/2) + 0.3*xi_j,
xi_j ~ N(0,I_d).
```

不同 pair 之间没有耦合；该测试用于观察高维、狭长弯曲 valley 下的表现。

### 3. Rastrigin with shifted initialization

代码名为 `rastrigin`，`A=10`，实际目标函数为

```text
Phi(theta) = sum_i [theta_i^2 + 20*sin(pi*theta_i)^2].
```

全局极小点是 `theta_star=0`，最优值为 0。每个坐标独立初始化为

```text
theta_(0,i)^(j) ~ Uniform[-3,5].
```

该分布的期望为 1，因此 ensemble mean 不会因为关于 0 对称而恰好落在全局极小点。“shifted”
仅指初始化区间发生了平移；Rastrigin 目标函数本身没有平移，也没有旋转。

### 4. Rotated Rastrigin

令 `B` 为由 Gaussian random matrix 的 QR 分解生成的正交矩阵，`z=B*theta`：

```text
Phi(theta) = sum_i [z_i^2 + 20*sin(pi*z_i)^2].
```

全局极小点仍为 `theta_star=0`，但旋转使目标函数在 `theta` 坐标下不再 separable。初始化与上一个
问题相同，每个坐标独立服从 `Uniform[-3,5]`。旋转矩阵 `B` 由该 trial 的 problem seed 生成，
因此不同 trial 使用不同的旋转实例。

## 随机种子与公共实验设置

当前两张结果表都使用

```text
d = 100
J = 50 或 20
maximum forward evaluations = 10000
Delta_tau = 0.5
dropout rate = 0.5
joint dropout weight = 1
trial seeds = 2026091550, ..., 2026091554
```

对 trial seed `s`，为避免增加或调整测试问题时改变其他问题的随机数，代码使用固定 problem-seed
offset：

| 问题 | problem seed | 本次实际范围 |
|---|---:|---:|
| monotone cubic | `s+1000` | `2026092550:2026092554` |
| paired Rosenbrock | `s+2000` | `2026093550:2026093554` |
| Rastrigin with shifted initialization | `s+3000` | `2026094550:2026094554` |
| rotated Rastrigin | `s+4000` | `2026095550:2026095554` |

测试函数中的随机对象使用 `MersenneTwister(problem_seed)`；初始 ensemble 使用独立的
`MersenneTwister(problem_seed+17)`。同一个问题和 trial 内，四种 ensemble 方法共享完全相同的
初始 ensemble。Inflation EAKI 使用 `problem_seed`，Sequential、Joint、DEKI 和 CMA-ES 使用
`problem_seed+1` 作为各自算法的随机种子；其中相同整数只保证每个方法自身可复现，并不表示
Julia 和 Python CMA-ES 使用相同的随机数流。输出 CSV 的 `seed` 列记录的是 trial seed `s`，
不是加 offset 后的 problem seed。

## 评估成本

横轴统计真实 forward-map evaluations；initial ensemble 另需一次性 `J` 次。

| 方法 | 每步 evaluations |
|---|---:|
| Inflation EAKI | `J+1` |
| Sequential projected dropout EAKI | `2J+2` |
| Joint projected EAKI | `2J+1` |
| DEKI | `2J+1` |
| CMA-ES | 每代 `J` |

ensemble 方法共享初始 ensemble。CMA-ES 从该 ensemble 的 sample mean 开始，`sigma0` 取初始
各坐标 sample variance 的平均平方根，population size 等于 `J`。主指标是所有实际查询点中的
best-so-far objective。

## `d=100, J=50, 10000 evaluations, 5 seeds`

best evaluated objective median：

| 问题 | Inflation EAKI | Sequential | Joint | DEKI | pycma |
|---|---:|---:|---:|---:|---:|
| monotone cubic | 805.977 | 1.798 | **1.677** | 103.714 | 107.188 |
| paired Rosenbrock | 203.417 | 89.490 | **85.904** | 210.120 | 169.799 |
| shifted Rastrigin | 341.686 | 176.108 | **139.294** | 228.133 | 818.864 |
| rotated Rastrigin | 350.279 | 201.976 | **156.208** | 285.217 | 835.497 |

Joint 相比 sequential projected 在 monotone、paired Rosenbrock、shifted Rastrigin、
rotated Rastrigin 上分别降低约 6.7%、4.0%、20.9%、22.7%。

## `d=100, J=20, 10000 evaluations, 5 seeds`

| 问题 | Inflation EAKI | Sequential | Joint | DEKI | pycma |
|---|---:|---:|---:|---:|---:|
| monotone cubic | 2383.276 | 19.095 | **17.438** | 804.570 | 28.907 |
| paired Rosenbrock | 938.077 | 196.346 | 196.155 | 375.802 | **140.359** |
| shifted Rastrigin | 642.095 | 186.058 | **178.098** | 613.134 | 211.464 |
| rotated Rastrigin | 669.628 | 162.178 | **146.259** | 592.753 | 250.358 |

Joint 相比 sequential projected 在 monotone、paired Rosenbrock、shifted Rastrigin、
rotated Rastrigin 上分别降低约 8.7%、0.1%、4.3%、9.8%。Paired Rosenbrock 在小集合
`J=20` 下是一个重要反例：Joint 和 Sequential 都明显优于 Inflation EAKI、DEKI，但不及
CMA-ES。5 seeds 只用于确认趋势，正式论文建议至少 10--20 seeds。

曾测试过相对 EAKI mean 的严格 safeguard。其效果随问题和 `J` 改变，而且每步至少增加两次
forward evaluations。为保持算法和论述简洁，最终代码不保留该分支；历史消融结果仍保存在
`Results/joint_safeguard_j20` 和 `Results/joint_safeguard_j50`。

## 运行

回归测试：

```bash
/Applications/Julia-1.13.app/Contents/Resources/julia/bin/julia \
  --startup-file=no TestInflationEKI/runtests.jl
```

主比较：

```bash
OPENBLAS_NUM_THREADS=1 \
  FUNCTION_NAMES=monotone_cubic,paired_rosenbrock,rastrigin,rotated_rastrigin \
  THETA_DIM=100 N_ENS=50 MAX_EVAL=10000 N_SEEDS=5 SEED=2026091550 \
  /Applications/Julia-1.13.app/Contents/Resources/julia/bin/julia \
  --startup-file=no TestInflationEKI/RunNonlinearSeedStudy.jl
```

脚本输出逐 seed CSV、median/IQR summary CSV、serialized trajectories 和 median/IQR 曲线图。

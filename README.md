# AdaSpeVLA 真机训练代码

整理日期：2026-11-10。
用途：在 **insert-tube** 和 **wipe-whiteboard** 两个任务上训练 AdaSpeVLA，并和各自的 π0.5 基线做对照。

不包含：检查点、数据集、模型权重、HF token / SSH key 等任何凭证。

## 目录

| 路径 | 内容 |
|---|---|
| `adaspe_train/adaspe/` | plugin 本体：`model.py`（AdaSpePi05 模型）、`bspline.py`（B 样条、knot、timing 标签、位姿几何）、`timing_data.py`（把未来 13 帧实测 state 带进 batch） |
| `adaspe_train/configs.py` | 训练 config（不改 OpenPI 源码，通过注册接入）。`TASK` 决定 config 名，如 `bread_adaspe_action`、`bread_adaspe_timing_from20k` |
| `adaspe_train/train.py`、`adaspe_train/compute_norm_stats.py` | OpenPI 脚本的包装，先注册 config 再运行 |
| `adaspe_train/calibrate_timing_anchor.py` | 标定 knot / timing 标签用的 `motion_scale` 和 `e_crit` |
| `adaspe_train/run_train.sh` | 全流程：预检 → Action 40k → 3 个 timing |
| `adaspe_train/run_timing_one.sh` | 单独跑一个 timing，便于多卡并行 |
| `adaspe_train/strip_old_train_state.sh` | 磁盘不够时用：删掉旧检查点里的优化器状态，只留参数 |
| `adaspe_train/wait_gpus_and_launch.sh` | 等 GPU 空闲再启动 |
| `adaspe_train/tests/` | 数值等价测试、冒烟测试、数据检查、标签检查 |
| `adaspe_train/assets/` | bread 的归一化统计（示例），以及标签检查用的参考文件 |
| `env_patches/` | 需要放进 venv site-packages 的补丁（见下文） |
| `reference_adaspevla_source/` | 原版 AdaSpeVLA PyTorch 代码，`tests/test_bspline_parity.py` 用它做对照。**不在公开仓库里**，需要自己放到这里，或用环境变量 `ADASPEVLA_REF` 指定位置 |

## Recipe

- OpenPI JAX，commit `215abfb`，不改上游源码。
- 骨干 PaliGemma 从官方 `pi05_base` 加载，全量微调；不加载 π0.5 action expert。plugin 头随机初始化，seed 0。
- 数据配置：OpenPI `LeRobotLiberoDataConfig`，`prompt_from_task=True`；state 按 π0.5 默认离散化写进 prompt；图像 `resize_with_pad` 到 224，使用默认增强。
- 归一化统计直接用基线模型仓库里的 `norm_stats.json`（分位数归一化），用前先重算一遍，确认一致。
- `action_horizon=13`（1.2s @10Hz），三次 B 样条 K=8（端点固定，ridge λ=1e-4），查询间隔 0.2–0.8s。
- flow 头：隐藏层 384，6 层，6 头，cond_dim 512，ODE 15 步。flow 的目标是归一化后的原始动作。
- knot 密度目标和 timing 标签都按原版 AdaSpeVLA 计算：
  - 运动来自窗口内的实测位姿（相对窗口第一帧，位置差 + 旋转向量），按 `motion_scale` 归一化；
  - 夹爪用实测开度，以 0.5 为界判断切换；
  - timing 标签是第一次运动变化量超过 `e_crit` 或夹爪切换的时间，裁剪到 [0.2, 0.8]s。
- 损失：Action 阶段 flow MSE + 0.5 × knot KL；timing 阶段 log-MSE。
- global batch 32，AdamW（b1 0.9，b2 0.95，wd 1e-10，梯度裁剪 1.0），cosine（warmup 1k，peak 2.5e-5，decay 30k，end 2.5e-6），EMA 0.99，评测用 EMA 参数。
- Action 训练 40k 步，保留 10k/20k/30k/40k。然后基于 20k/30k/40k 各训练一个 timing 头：10k 步，只训练 `freq_head`，其余全部冻结。
- 每 1000 步存一次检查点，wandb 关闭，FSDP 设备数等于 GPU 数。

## 环境搭建

1. 克隆 OpenPI 并切到 `215abfb`，`uv sync --frozen --python 3.11`。缺内核头文件时加 `--no-install-package evdev`。
2. 把 `env_patches/` 里的 `datasets_list_shim.py` 和 `datasets_list_shim.pth` 放进 `external/openpi/.venv/lib/python3.11/site-packages/`：让 datasets 3.6 认识 `List` 特征（训练、norm stats 和检查脚本都要用）。不要把 datasets 升级到 4.x。
   仓库里的脚本自己会找到 `adaspe_train/`，不需要别的路径补丁。只有在仓库外的代码里 `import configs`（比如自己写的 policy server）时，才需要在仓库根目录运行一次：
   `echo "$PWD/adaspe_train" > external/openpi/.venv/lib/python3.11/site-packages/adaspe_train_path.pth`
3. PaliGemma tokenizer 要预先下载到 HF cache。脚本默认离线运行（`HF_HUB_OFFLINE=1`）。

## 路径

代码里没有写死的路径：所有脚本都从自己所在的位置推出仓库根目录（Python 用 `__file__`，shell 用 `BASH_SOURCE`），仓库放在哪里都可以，不用改路径。

根目录（即本仓库）下应有的布局，每一项都可以是软链接（比如指到更大的磁盘）：
- `external/openpi`：OpenPI 和 venv
- `adaspe_train`：这份代码
- `reference_adaspevla_source`：原版 AdaSpeVLA 代码（数值等价测试用，不在公开仓库里；也可以用环境变量 `ADASPEVLA_REF` 指定别的位置）
- `data/lerobot/<repo_id>`：LeRobot 数据集
- `checkpoints/`：训练输出
- `test/franka/<实验名>/`：日志
- `.cache/`：HF、JAX 等缓存

`pi05_base` 参数默认在 `.cache/openpi/openpi-assets/checkpoints/pi05_base/params`（`configs.py` 的 `PI05_BASE`）。

## 换成新任务要改的地方

两个任务的格式和 bread 一致：两路 128×128 图像，19 维 state，7 维动作，10Hz，各只有一条 task 文本。

| 任务 | 数据集（revision） | 基线模型（revision） | 基线 norm_stats 在仓库里的路径 |
|---|---|---|---|
| insert-tube | `dddddsds31341/insert-tube-lerobot`（`70bcd1f86ed35fc93a0422a564472355242a51ab`），102 条 / 16,403 帧 | `dddddsds31341/Franka-Insert-Tube-pi05-step-20000`（`e97d90f5bed133b24e8de4189eb25254641e69cf`） | `assets/bofang/insert-tube-lerobot/norm_stats.json` |
| wipe-whiteboard | `dddddsds31341/wipe-whiteboard-lerobot`（`2c8cfa8ddef1f83cb09ca7881a0944cd637cd6d0`），112 条 / 29,795 帧 | `dddddsds31341/Franka-Wipe-Whiteboard-pi05-step-20000`（`628df08d8cea3965d8a5ece6f57a8f60dd242c2e`） | `assets/bofang/wipe-whiteboard-lerobot/norm_stats.json` |

1. **configs.py**
   - `REPO_ID` 改成新数据集。
   - 把基线的 `norm_stats.json` 放到 `adaspe_train/assets/baseline/<REPO_ID>/norm_stats.json`。注意基线仓库里的目录名是 `bofang/...`，要按我们的 `REPO_ID` 放。
   - `TASK` 改成新任务名（如 `insert_tube`），config 名会跟着变。检查点放在 `checkpoints/adaspe/<TASK>_adaspe_*`，不会和别的任务混在一起。
2. **重新标定，不能沿用 bread 的数**（只读数据，CPU 上运行）：
   - 改 `calibrate_timing_anchor.py` 里的 parquet 路径，然后运行。
   - 把输出的 `motion_scale` 和 `e_crit` 填进 `configs.py` 的 `TIMING_MOTION_SCALE` 和 `TIMING_E_CRIT`。
   - 记录标签分布，即 0.2s、0.8s 各占多少。
3. **核对 state 含义**。代码假设：
   - `state[0]` 是夹爪开度（开 1，合 0）；
   - `state[4:7]` 是位置；
   - `state[7:10]` 是 Euler xyz（scipy "xyz"）。

   先用 `tests/check_data.py` 和几条 episode 确认新数据也是这样。wipe-whiteboard 可能全程拿着板擦，夹爪不开合；这不影响训练，只是 timing 事件全部来自运动。
4. **run 脚本**：运行时设置环境变量 `TASK`，要和 `configs.py` 里的 `TASK` 一致。日志默认写到 `test/franka/<TASK>_adaspe/`。
5. `tests/check_timing_labels.py` 依赖 `assets/timing_labels/<TASK>_H<H>.npy`。先用新任务的数据运行 `calibrate_timing_anchor.py` 重新生成这个文件，再跑检查。

## 运行顺序

1. 重算 norm stats，和基线逐位比较：`python adaspe_train/compute_norm_stats.py --config-name pi05_<TASK>_franka_ref`
2. 标定（见上文）。
3. CPU 测试：
   - `tests/test_bspline_parity.py`：和原版的误差 ≤1e-12；
   - `tests/smoke_model.py`；
   - `tests/check_data.py`；
   - `tests/check_timing_labels.py`：0 处不一致。
4. 预检和正式训练：`CUDA_VISIBLE_DEVICES=<卡> TASK=<任务名> bash adaspe_train/run_train.sh`。可以先设 `PREFLIGHT_ONLY=1`，只跑预检：训练 30 步，再从断点续训，确认 loss 有限、续训正常。
5. timing 想并行的话，Action 跑完后每个 timing 用一张卡：`CUDA_VISIBLE_DEVICES=4 TASK=<任务名> bash adaspe_train/run_timing_one.sh 20`，30、40 同理。

注意：Action 开始后不要改卡数。检查点只能在保存时的同样卡数上续训；global batch 32 要能被卡数整除，所以用 1、2 或 4 张卡。

## 交付时请给我

- 最终的 Action（20k/30k/40k）和 timing EMA 参数，以及所有文件的 sha256；
- loss 曲线；
- 实际用的 config 导出、标定出的 `motion_scale` / `e_crit`、标签分布、git commit；
- 和本文档不一致的地方，逐条列出。

## Jean Zay 上的配置（2026-11-10）

只在 Jean Zay 上用。别的机器不需要 `adaspe_train/slurm/`，直接用 `run_train.sh` / `run_timing_one.sh`。

- WORK 的 inode 配额紧张，所以 `external/`（OpenPI `215abfb` + uv venv）、`data/`、`checkpoints/`、`.cache/` 都是指向 `$SCRATCH/adaspevla_train/` 的软链接。SCRATCH 上 30 天不访问的文件会被清掉，交付用的参数要拷走。
- 任务由环境变量 `TASK` 选择（`bread` / `insert_tube` / `wipe_whiteboard`），各任务的 `REPO_ID`、`motion_scale`、`e_crit` 写在 `adaspe_train/tasks.py`；`configs.py`、标定脚本和测试都从这里读，不用再手改 `configs.py`。还没标定的任务只注册 `pi05_<TASK>_franka_ref`。
- 标定结果：`assets/calibration/<TASK>_H<H>.json`；标签文件：`assets/timing_labels/<TASK>_H<H>.npy`（原 bread 的 `timing_labels_measured.npy` 保留不动，是 H=13 旧 recipe 的）。标定命令：`TASK=<任务> python adaspe_train/calibrate_timing_anchor.py`，结果要抄进 `tasks.py`（标定和 check 用的都是 `adaspe/labels.py` 里和训练相同的标签函数）。
- `pi05_base`：`.cache/openpi/openpi-assets/checkpoints/pi05_base/params`（官方 GCS，与 bucket 列表逐文件核对过大小）。tokenizer 在 `.cache/openpi/big_vision/paligemma_tokenizer.model`（OpenPI 从 `OPENPI_DATA_HOME` 读，不是 HF cache）。
- 计算节点不联网，所有数据/权重都在本地；脚本里 `unset PYTHONPATH LD_LIBRARY_PATH`、`PYTHONNOUSERSITE=1`，避免模块和 `~/.local` 混进 venv。
- SLURM（`adaspe_train/slurm/`）：都要在仓库根目录提交（SLURM 运行的是脚本的副本，根目录和日志路径都取提交目录），账号用 `-A <账号>` 给，脚本里不写账号。
  - `prep_checks.slurm`：CPU 上跑 norm stats 重算+比较、parity、smoke、check_data、check_timing_labels（两个任务）。
  - `preflight.slurm`：H100 dev，4 卡单独跑 preflight，作业名必须是 `adaspe-pf-<TASK>`，写 `preflight_passed`（可选：dev QoS 全集群共用 128 卡，常排不上；`train_task.slurm` 第一节自己会先跑 preflight，失败就停链）。
  - `train_task.slurm`：H100 t3 20 h 一节的链式作业（singleton，自动排下一节，最后一节或真失败时取消）。4 卡 FSDP 4 跑 Action 40k，然后 3 个 timing 各占一张卡并行。
    `sbatch -A <账号> --job-name=adaspe-insert_tube --export=ALL,TASK=insert_tube adaspe_train/slurm/train_task.slurm`（后面各节自动用同一个账号）
  - 日志：`test/franka/<作业名>-<id>.out` 和 `test/franka/<TASK>_adaspe/`。

## Recipe 修订（2026-11-10，和上面 "Recipe" 不同的地方）

开关都在 `adaspe_train/tasks.py`，方括号里是原值；改回原值并重新标定即可复现原 recipe（`adaspe/labels.py` 在原设置下和旧标签逐个一致，0 处不一致）。

| 项 | 现在 | 原来 | 原因（两个任务的数据） |
|---|---|---|---|
| `ACTION_HORIZON` | 16（1.5 s） | 13（1.2 s） | 13 没有必须的理由；16 拟合误差配 K=11 反而更小 |
| `NUM_CONTROL_POINTS` | 11 | 8 | 归一化动作的位姿 RMSE 0.024/0.021（原 0.029/0.025） |
| query interval | 0.2–1.5 s（上限 = horizon） | 0.2–0.8 s | 原上限 72–76% 的标签被顶住，并且强制截掉 chunk 后 1/3 |
| 夹爪事件（timing 标签、knot 密度） | 夹爪**命令**的开/合状态 | 实测开度过 0.5 | 实测开度比命令晚中位 19 帧（~1.9 s，前 ~1.4 s 完全不动，机械臂静止），超过 chunk 长度 |
| knot 密度 | `1 + 10·tanh(mv / e_crit)` | `1 + 10·tanh(mv)` | 原公式 tanh 在 68–73% 的帧间隔饱和，62–63% 的窗口 knot 目标近似均匀；现在 3–4% |
| 推理 `sample_spline` | 返回 `execute_steps` 和 `dense_actions`：前缀至少执行到预测的夹爪切换 | 只有 query interval | 静止时下一帧观测不变，切换若落在前缀之后会被一直推后 |

标定（H=16）：insert-tube `e_crit` 11.98，标签 0.2 s 12.0% / 1.5 s 54.6%；wipe-whiteboard `e_crit` 8.96，0.2 s 10.2% / 1.5 s 60.0%。
在开爪命令处，样条在命令那一帧的值中位数 +0.93 / +0.98（均匀 knot 时 +0.66）。

另外修了一个原代码里的 bug：`configs.py` 里 `import openpi.models.model as _model` 被后面的 `def _model(stage)` 覆盖，两个 weight loader 调 `_model.restore_params` 会在训练开始时报 `AttributeError`。现在模块别名是 `_openpi_model`，config 函数叫 `model_config`。

部署（不在本仓库）：客户端调用 `sample_spline`，执行 `dense_actions[:, :execute_steps]`（每步一个 0.1 s 的增量命令），然后重新观测、推理。

## Timing 标签怎么算（当前 recipe，H=16）

timing 头预测**查询间隔** q ∈ [0.2, 1.5] s：当前 chunk 执行多久之后重新观测、推理。标签的意思是"离下一个事件还有多久"：快有事发生（急加减速、夹爪开合）就早点重新推理，动作平稳就把整个 chunk 执行完。

对每个训练帧 t，取未来 16 帧 t … t+15（dt = 0.1 s）的**实测** state 窗口。episode 末尾不够 16 帧时重复最后一帧（LeRobot 的 clamp，相当于停在终点）：

1. **相对位姿**：实测末端位姿 `state[4:10]`（xyz + Euler xyz，scipy `"xyz"`）换成相对第 t 帧的偏移：位置 `p_j − p_t`，旋转 `Log(R_j R_tᵀ)`（旋转向量），共 6 维。每维除以 `motion_scale`，即该维 |偏移| 在全数据集上的 95 分位数，得到无量纲的运动量。
2. **运动变化量**：差分求速度 `v = Δ偏移 / dt`，再求加速度的模 `a_i = ‖Δv / dt‖`。`a_i` 用到第 i、i+1、i+2 帧，记在第 i+1 帧（时间 0.1·(i+1) s）；最后一个间隔复制前一个。见 `bspline.motion_variation`。
3. **事件**：
   - 运动事件：`a_i ≥ e_crit`。`e_crit` 是 `a` 在全数据集上的 95 分位数，即数据里最剧烈的 5% 加速度（insert-tube 11.98，wipe-whiteboard 8.96）。
   - 夹爪事件：夹爪**命令**的开/合状态在相邻两帧之间切换。命令是 −1 合 / +1 开 / 0 不变；0 沿用前一个非零命令，窗口开头没有非零命令时用后一个，全是 0 就不产生事件。不用实测开度，因为开度比命令晚约 1.9 s（见上表）。
4. **标签** `y` = 第一个事件的时间（0.1, 0.2, …, 1.5 s），裁剪到 [0.2, 1.5]；窗口内没有事件时 `y = 1.5`。例如机械臂平稳移动，0.7 s 后为抓取急刹，则 `y = 0.7`。

标签分布：insert-tube 0.2 s 12.0%、1.5 s 54.6%，其余在 0.3–1.4 s 之间大致均匀；wipe-whiteboard 0.2 s 10.2%、1.5 s 60.0%。

实现：标签在训练时由模型在线计算（`adaspe/labels.py` 的 `timing_target` → `event_inputs`，`adaspe/bspline.py` 的 `query_interval_target`、`motion_variation`、`measured_anchor_offsets`）。实测窗口由 `adaspe/timing_data.py` 写进 action 的补零通道 7–13；flow 头和 knot 头不读这些通道。`motion_scale` / `e_crit` 由 `calibrate_timing_anchor.py` 标定（写进 `tasks.py`），它用同一套标签函数生成 `assets/timing_labels/<TASK>_H16.npy`，`tests/check_timing_labels.py` 对比训练管线的标签：0 处不一致。knot 密度目标用的是同样的归一化运动量和夹爪命令（`labels.knot_span_target`）。

**timing 头和损失**（`adaspe/model.py` 的 `timing_interval`、`compute_loss` 里 `stage == "timing"` 的分支）：
- 输入（全部 stop-gradient）：pooled 观测特征 `cond`（512）、log horizon、8 个 knot 间隔、预测样条在 16 个时刻的值 ×7（6 维运动 + 夹爪开/合）。共 633 维，经过 MLP 633 → 128 → 1 得到 z，`q = 0.2 + 1.3·sigmoid(z)`。
- 损失：`(log q − log y)²` 的 batch 平均。在 log 空间算，误差是相对的：0.2 预测成 0.4 和 1.5 预测成 0.75 的代价一样。损失 L 对应典型误差倍数 `exp(√L)`。常数预测的损失 = Var(log y) ≈ 0.52（insert-tube）/ 0.47（wipe-whiteboard），即约 2.1 倍。
- 难点：标签来自未来的**实测**加速度（接触、刹车、示教里的抖动），当前图像不能完全决定它；而头只能看到模型预测的命令样条，不是实测轨迹。所以损失有一部分降不下去。
- 推理：执行 `ceil(q / 0.1)` 步；如果 chunk 里预测有夹爪切换，至少执行到切换之后（`sample_spline`）。

学习率（2026-11-14）：原 recipe 的 timing 阶段沿用 OpenPI 默认 LR（peak 2.5e-5，decay 30k），但只训练 10k 步，所以 LR 一直在 2.0–2.5e-5，对一个新初始化的小 MLP 偏小。新加的 config `<TASK>_adaspe[_expert]_timing_from40k_lr{3e-4,1e-3}`（`configs.TIMINGS_LR`，`slurm/timing_lr.slurm`）同样 10k 步、warmup 1k，cosine 衰减到 peak/10。按 episode 留出 20% 做验证（`adaspe_train/lr_check.py`，在 `dump_timing_features.py` 导出的特征上离线重放 timing 阶段），insert-tube 第 10k 步的留出损失：

| LR | flow_head | expert |
|---|---|---|
| 2.5e-5（原） | 0.217 | 0.292 |
| **3e-4** | **0.195** | **0.270** |
| 1e-3 | 0.216（过拟合，训练损失 0.088） | 0.265（训练损失 0.089） |

部署用 `lr3e-4`。wipe-whiteboard 的留出验证还没做。

## 变体：pi0.5 action expert 生成控制点（2026-11-13，`VARIANT=expert`）

动机：真机上 π0.5 基线的精度明显好于 AdaSpeVLA。两者的最大差别在生成动作的部分。AdaSpeVLA 用 20.4 M 的 flow 头，从零训练，只看 backbone 最后一层投影后的 token；π0.5 的 Gemma-300M expert 有 311 M transformer 参数，另有 117 M adaRMS 时间调制层，在 `pi05_base` 里预训练过，每一层都 attend 到 backbone 的 KV。原插件加载时把它丢掉了。样条本身不是瓶颈：用本 recipe 的样条（H=16、K=11）拟合 demo，9 步前缀累计误差的 p95 ≤ 一个典型步长的 5%（慢速阶段也一样）。

做法（`adaspe/model.py` 的 `generator="pi05_expert"`；默认的 `"flow_head"` 代码和参数初始化都不变，旧检查点照常用）：
- 网络是完整的 π0.5（PaliGemma + action expert），`pi05_base` 的全部权重都加载。`PluginWeightLoader` 会检查 expert 和 `action_in_proj` / `action_out_proj` / `time_mlp_*` 是否齐全、形状是否一致，缺一个就报错。
- expert 的输入是 K=11 个控制点 token，π0.5 原本是 action token。token = `action_in_proj(带噪控制点，按 OpenPI 补零到 32 维)` + `cp_knot_proj(该控制点 Greville 位置的 sin-cos 编码、knot 向量、log horizon)`。`cp_knot_proj` 初始化为 0，所以第 0 步 expert 收到的输入和 π0.5 完全一样。
- flow 按 π0.5 的约定：t=1 是噪声，时间采样 Beta(1.5, 1)，ODE 10 步（timing 阶段取特征也用 10 步）。adaRMS 时间调制、attention mask、位置编码和 KV cache 的用法都照搬 `Pi0`。
- knot 要从 prefix 预测，所以先单独跑一遍 prefix 拿 KV cache，再跑 expert（和 π0.5 推理时一样）。梯度穿过 KV cache，所以训练等价于 π0.5 的 prefix+suffix 联合前向（`tests/test_expert_parity.py` 有对比）。
- 不变的部分：knot 头、timing 头、标签、样条拟合、损失（flow MSE + 0.5 × knot KL）、数据、norm stats、40k 步、保留 10k/20k/30k/40k、timing 阶段。

运行：和原变体一样，`--export` 里加 `VARIANT=expert`，作业名前缀用 `adaspe-expert-`。原变体是 `adaspe-`，两个变体的作业链互不干扰：
```
sbatch -A <账号> --qos=qos_gpu_h100-dev --time=02:00:00 --job-name=adaspe-expert-dev-insert_tube \
    --export=ALL,TASK=insert_tube,VARIANT=expert adaspe_train/slurm/train_task.slurm
```
config：`<TASK>_adaspe_expert_action`、`<TASK>_adaspe_expert_timing_from{20,30,40}k`。日志在 `test/franka/<TASK>_adaspe_expert/`，检查点在 `checkpoints/adaspe/<TASK>_adaspe_expert_*`。名字统一由 `adaspe_train/names.sh` 生成。
测试（CPU）：`tests/test_expert_parity.py`、`tests/smoke_model.py`（两种 generator 都测）、`tests/check_config_names.py`，都在 `slurm/prep_checks.slurm` 里。

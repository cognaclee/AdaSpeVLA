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
| `reference_adaspevla_source/` | 原版 AdaSpeVLA PyTorch 代码，`tests/test_bspline_parity.py` 用它做对照 |

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
- `reference_adaspevla_source`：原版 AdaSpeVLA 代码（数值等价测试用；也可以用环境变量 `ADASPEVLA_REF` 指定别的位置）
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

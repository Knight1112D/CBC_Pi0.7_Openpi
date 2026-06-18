# 尝试将 RTC + PI 系列模型结合复现，控制人形机器人在真机作出合理动作
---
## 计划
- 1. 先使用这个项目的 pi 0.5 版本 让其可以训练 天工机器人 双手任务
- 2. 处理数据集转换成合适的 lerobot 格式
- 3. 实现异步推理
- 4. 尝试改造代码 在推理的部分 改造命令修正项 适配rtc
- 5. 尝试改造代码成为pi 0.6 的融合 recap rl 的版本

## 进度

- 1. 在下面路径，定义了 `TienkungInputs` 和 `TienkungOutputs` 注意转换的数据集命名要匹配这里。

```text
/data/caobochun/openpi/src/openpi/policies/tienkung_dual_hands_policy.py
```

- 2. 在下面路径，定义了数据转换的配置类 `LeRobotTienkungDataConfig` 这里模型是增量控制 绝对位值指令要转换成增量，在这里定义

```text
/data/caobochun/openpi/src/openpi/training/config.py
```

- 3. 在上面路径同时定义了 `TrainConfig` 里面 `pi05_tienkung_finetune`

- 4. 当前天工双手任务的数据和控制维度按真实采集数据的 26 维处理，顺序约定如下。

```text
左臂 7 维 + 左手 6 维 + 右臂 7 维 + 右手 6 维 = 26 维
```

其中双手的 6 维暂时按五指合并后的单手 6 自由度处理。模型本身 `pi05_base` 仍然是 32 维 state/action，因此天工实际 26 维会在 transform 里补齐到 32 维，推理输出时再裁剪回前 26 维。

- 5. `TienkungInputs` / `TienkungOutputs` 当前状态：

```text
/data/caobochun/openpi/src/openpi/policies/tienkung_dual_hands_policy.py
```

这里负责把三路图像和 26 维状态转换成 pi0.5 模型需要的输入格式：

```text
observation/base_image  -> base_0_rgb
observation/left_image  -> left_wrist_0_rgb
observation/right_image -> right_wrist_0_rgb
observation/state       -> state
prompt                  -> prompt
```

推理输出时只取模型动作的前 26 维作为天工双手动作。

- 6. `LeRobotTienkungDataConfig` 当前状态：

```text
/data/caobochun/openpi/src/openpi/training/config.py
```

这里定义了 LeRobot 数据字段到训练字段的 repack 映射，并开启了绝对动作转增量动作：

```text
delta_action_mask = make_bool_mask(7, -6, 7, -6)
```

也就是左右臂 7 维关节使用增量动作，左右手各 6 维保持绝对值。这个和目前“机器人发送/采集的是绝对位值，pi0.5 训练使用增量动作”的设定一致。

- 7. `pi05_tienkung_finetune` 当前状态：

```text
model=Pi0Config(pi05=True)
```

当前使用 pi0.5 默认 `action_horizon=50`。`action_dim` 没有改成 26，而是保持 pi0.5 默认的 32，避免和本地 `pi05_base` checkpoint 不匹配。天工真实 26 维通过 transform 补齐到 32，输出再裁剪回前 26 维。

权重加载当前按 PyTorch safetensors 路线处理：

```text
pytorch_weight_path="/data/caobochun/openpi/checkpoints/pi05_base"
weight_loader=NoOpWeightLoader()
num_train_steps=10_000
```

因此后续训练应优先走：

```bash
uv run scripts/train_pytorch.py pi05_tienkung_finetune --exp_name <实验名>
```

不要直接走 JAX 的 `scripts/train.py`，除非之后另外准备 JAX `params` 格式的 pi05 checkpoint。

- 8. 当前真实 LeRobot 数据集已经转换完成，repo id 和本地 root 如下：

```text
repo_id="caobochun/tienkung_dual_hands_take_box_13_26d"
lerobot_root="/data/caobochun/openpi/data/lerobot/caobochun/tienkung_dual_hands_take_box_13_26d"
```

数据来自 `/data/caobochun/openpi/data/Tienkung_dual_hands_take_box_14`，其中 `0000` 到 `0012` 共 13 组用于训练，`0013` 留作验证。已经计算 norm stats，路径如下：

```text
/data/caobochun/openpi/assets/pi05_tienkung_finetune/caobochun/tienkung_dual_hands_take_box_13_26d/norm_stats.json
```

- 9. 当前已经在下面目录准备了天工双手真机部署客户端：

```text
/data/caobochun/openpi/examples/tienkung_dual_hands
```

目前保留的文件为：

```text
openpi_dual_hands_client.py
tienkung_dual_hands_config.example.json
async_policy.py
ros_io.py
tienkung_config.py
trajectory.py
```

- 10. `examples/tienkung_dual_hands` 已经改成异步推理结构，不再把所有逻辑都堆在一个脚本里。

```text
openpi_dual_hands_client.py              # 入口，20Hz 控制循环和异步调度
async_policy.py                          # 独立进程连接 policy server 做推理
ros_io.py                                # ROS2 订阅图像/状态和发布电机命令
tienkung_config.py                       # 配置读取、电机布局、限幅
trajectory.py                            # action chunk 插值到控制频率
rtc_chunker.py                           # RTC 调度层：延迟估计、动态 horizon、suffix 软融合
convert_tienkung_data_to_lerobot.py      # 原始数据转换为 LeRobot
eval_tienkung_holdout.py                 # 离线验证留出/训练 episode 的 action chunk 输出
```

当前结构是控制进程和推理进程分离：主进程保持 20Hz 控制循环，不等待模型推理；推理进程通过 websocket 访问 policy server，返回 action chunk 后主进程把 chunk 从 `policy_action_hz` 插值到 `control_hz`，再取前 `open_loop_horizon` 步放入执行队列。

目前 example 配置里默认：

```text
control_hz=20
policy_action_hz=20
interpolation="linear"
open_loop_horizon=8
request_when_remaining_steps=10
max_action_chunk_len=32
```

这只是部署侧第一版默认值，最终控制区间和重规划频率仍需要真机调试时确认。

- 11. 当前部署侧有两个控制模式，可以通过配置文件的 `controller_mode` 选择：

```text
controller_mode="async"  # 普通异步推理：队列快耗尽时请求新 chunk，返回后直接替换执行队列
controller_mode="rtc"    # RTC 调度验证：动态执行 horizon、延迟 buffer、保留上一 chunk suffix 并软融合
```

RTC 版本分为两层：

```text
1. 部署侧 RTC 调度层：延迟估计、动态 horizon、上一 chunk suffix 软融合。
2. 模型侧 RTC guidance：在 PyTorch flow matching 采样循环里对 v_t 加 ΠGDM 风格速度修正量。
```

模型侧速度修正已经做成可选路径，默认关闭，不影响普通推理。涉及文件：

```text
/data/caobochun/openpi/src/openpi/policies/policy.py
/data/caobochun/openpi/src/openpi/models_pytorch/pi0_pytorch.py
```

当部署侧请求里包含 `rtc_guidance` 时，`Policy.infer()` 会把部署侧的目标动作和 soft mask 经过同一套输入 transform 转换到模型采样空间，然后传给 `PI0Pytorch.sample_actions()`。模型内部在每个 denoising step 中由：

```text
x_t = x_t + dt * v_t
```

变为：

```text
x_t = x_t + dt * (v_t + guidance)
```

其中 guidance 通过 `A_hat_1 = x_t + (1 - tau) * v_t`、soft mask 加权误差和 `torch.autograd.grad` 计算 VJP 得到。没有传 `rtc_guidance` 时仍然走原来的 `torch.no_grad()` 路径。

当前 RTC 相关参数在配置文件中：

```text
rtc_min_horizon=8
rtc_delay_buffer_size=8
rtc_initial_delay_steps=8
rtc_blend_steps=6
rtc_soft_preserve_weight=0.85
rtc_model_guidance_enabled=false
rtc_guidance_beta=1.0
rtc_guidance_decay=0.9
rtc_guidance_eps=1e-4
```

RTC 模式的实际流程是：

```text
1. 20Hz 控制循环每步发布动作，不等待推理。
2. 当已执行步数达到 rtc_min_horizon 且没有推理在途时，提交最新观测。
3. 推理进程返回新 chunk 后，记录本次推理延迟步数。
4. 根据 delay buffer 估计下一次保守延迟。
5. 将上一 chunk 未执行 suffix 和新 chunk 前段软融合。
6. 切换到融合后的新 chunk 继续执行。
7. 如果 `rtc_model_guidance_enabled=true`，还会把上一 chunk 的未执行部分和 soft mask 发送到 policy server，在模型 flow matching 采样循环中加入速度修正量。
```

- 12. 机器人侧客户端不直接加载模型。它参考官方 `examples/aloha_real` 和 `examples/droid` 的方式，通过 websocket 连接 policy server，发送当前观测并接收 action chunk。

客户端当前支持的观测字段和训练侧保持一致：

```text
observation/base_image
observation/left_image
observation/right_image
observation/state
prompt
```

动作执行顺序同样固定为：

```text
左臂 7 + 左手 6 + 右臂 7 + 右手 6
```

本地机器人交互中不确定的内容已经放到配置文件里，包括 ROS 话题、三路相机话题、左右臂电机 ID、左右手电机 ID、限幅、控制频率、open-loop 执行步数、命令速度电流、policy server 地址等。

- 13. 当前还需要用户确认或补充的关键参数：

```text
1. 左手 6 维电机 ID
2. 右手 6 维电机 ID
3. 左右 d405 的真实 ROS 图像话题
4. 左右手 6 维的真实限幅范围
5. 部署侧 `policy_action_hz`、`max_action_chunk_len`、`open_loop_horizon` 是否和训练侧 50 步 action chunk 对齐
6. 普通异步推理时每次执行 action chunk 的前几步，当前 example 默认 open_loop_horizon=8
7. RTC 模式下 rtc_min_horizon、delay buffer、blend_steps、soft_preserve_weight 是否合适
8. 是否开启 rtc_model_guidance_enabled，以及 beta、decay、eps 的取值
9. 训练 batch_size、num_train_steps、wandb_enabled 等训练参数
10. 是否需要增加更多采集数据，改善留出集泛化
```

- 14. 当前建议的第一版训练参数倾向：

```text
action_dim=32
action_horizon=50
max_token_len=200
pytorch_training_precision="bfloat16"
batch_size=32
num_train_steps=10000
wandb_enabled=True，服务器未配置 wandb key 时用 `WANDB_MODE=offline`
```

当前训练已经按上面的第一版参数跑完一轮。预测区间、控制区间、重规划频率等仍属于重要 VLA/真机部署参数，需要在真机调试时继续确认。

## 2026-06-12 至 2026-06-15 实验记录：pi0.5 天工双手拿黑盒子任务

### 1. 数据和任务定义

- 原始数据目录：

```text
/data/caobochun/openpi/data/Tienkung_dual_hands_take_box_14
```

- 共 14 组 episode：

```text
0000 ~ 0012 作为训练集，共 13 组
0013 作为留出验证集，不参与训练
```

- 语言指令：

```text
Pick up the black box on the table with both hands, hold it briefly, then put the box down.
```

- 真实状态/动作维度：

```text
左臂 7 + 左手 6 + 右臂 7 + 右手 6 = 26
```

- 模型侧仍使用 pi0.5 默认 32 维 action/state，训练 transform 中自动 pad 到 32；输出 transform 中裁剪回前 26 维。

### 2. LeRobot 转换

- 转换脚本：

```text
/data/caobochun/openpi/examples/tienkung_dual_hands/convert_tienkung_data_to_lerobot.py
/data/cbc_skills/tienkung_vla_training/scripts/convert_tienkung_lerobot.sh
```

- 转换后的本地 LeRobot 数据集：

```text
/data/caobochun/openpi/data/lerobot/caobochun/tienkung_dual_hands_take_box_13_26d
```

- LeRobot repo id：

```text
caobochun/tienkung_dual_hands_take_box_13_26d
```

- fps 使用 20。当前口径是采集、训练和控制频率都按 20Hz 理解；action chunk 的真实执行时长还会受部署端 `max_action_chunk_len`、`open_loop_horizon`、插值和重规划策略影响。

- 图像在 LeRobot 中保留原始分辨率：

```text
base_image: 480x640
left_image/right_image: 240x424
```

进入模型前由 `ResizeImages(224, 224)` 统一转换为：

```text
base_0_rgb / left_wrist_0_rgb / right_wrist_0_rgb: 224x224
```

### 3. norm stats

- 已完成归一化统计：

```text
uv run scripts/compute_norm_stats.py --config-name pi05_tienkung_finetune
```

- 输出：

```text
/data/caobochun/openpi/assets/pi05_tienkung_finetune/caobochun/tienkung_dual_hands_take_box_13_26d/norm_stats.json
```

- checkpoint `10000` 内也保存了同一份 assets：

```text
/data/caobochun/openpi/checkpoints/pi05_tienkung_finetune/tienkung_take_box_26d_20fps_h50_full_10k/10000/assets/caobochun/tienkung_dual_hands_take_box_13_26d/norm_stats.json
```

### 4. 训练配置和启动

- 训练 config：

```text
pi05_tienkung_finetune
```

- 关键参数：

```text
model=Pi0Config(pi05=True)
action_dim=32
action_horizon=50
max_token_len=200
pytorch_training_precision=bfloat16
batch_size=32
num_train_steps=10000
optimizer=AdamW
lr_schedule=CosineDecaySchedule(warmup_steps=1000, peak_lr=2.5e-5, decay_steps=30000, decay_lr=2.5e-6)
extra_delta_transform=True
delta_action_mask=make_bool_mask(7, -6, 7, -6)
```

- 使用本地 pi0.5 PyTorch 权重：

```text
/data/caobochun/openpi/checkpoints/pi05_base/model.safetensors
```

- 训练使用 GPU 5 和 GPU 7：

```bash
CUDA_VISIBLE_DEVICES=5,7 WANDB_MODE=offline torchrun --standalone --nnodes=1 --nproc_per_node=2 scripts/train_pytorch.py pi05_tienkung_finetune --exp-name tienkung_take_box_26d_20fps_h50_full_10k --overwrite
```

- 训练时服务器没有配置 wandb API key，因此使用 `WANDB_MODE=offline`，wandb 本地记录保留。

- PyTorch 模型要求 `transformers_replace` 覆盖到 venv 的 transformers 包，已执行：

```bash
cp -r ./src/openpi/models_pytorch/transformers_replace/* .venv/lib/python3.11/site-packages/transformers/
```

检查结果：

```text
transformers_replace_ok True
```

### 5. 训练产物和清理

- 最终 checkpoint：

```text
/data/caobochun/openpi/checkpoints/pi05_tienkung_finetune/tienkung_take_box_26d_20fps_h50_full_10k/10000
```

- 该目录包含：

```text
model.safetensors
optimizer.pt
metadata.pt
assets/caobochun/tienkung_dual_hands_take_box_13_26d/norm_stats.json
```

- 中间 checkpoint `1000,2000,...,9000,9999` 已按用户要求删除，只保留 `10000`。实验 checkpoint 目录从约 215G 降到约 20G。

### 6. 留出集和训练集离线验证

- 新增离线验证脚本：

```text
/data/caobochun/openpi/examples/tienkung_dual_hands/eval_tienkung_holdout.py
```

- 脚本读取原始 episode 的三路图像和真实 state，加载训练后的 `10000` checkpoint，预测 50 步 action chunk，并和该 episode 中未来 50 步 command 做数值对比。

- 推理默认使用 flow matching 10 步：

```text
sample_kwargs={"num_steps": 10}
```

- 留出集 `0013` 输出目录：

```text
/data/caobochun/openpi/eval_outputs/tienkung_take_box_26d_20fps_h50_full_10k_holdout0013
```

留出集抽样结果：

```text
frame 0    mae_all26=0.000679  mae_arm14=0.001261  mae_hand12=0.000000
frame 51   mae_all26=0.049080  mae_arm14=0.091148  mae_hand12=0.000000
frame 102  mae_all26=0.035460  mae_arm14=0.065853  mae_hand12=0.000000
frame 154  mae_all26=0.037199  mae_arm14=0.069084  mae_hand12=0.000000
frame 205  mae_all26=0.028291  mae_arm14=0.052541  mae_hand12=0.000000
frame 257  mae_all26=0.016747  mae_arm14=0.031102  mae_hand12=0.000000
```

- 训练集 `0000` 输出目录：

```text
/data/caobochun/openpi/eval_outputs/tienkung_take_box_26d_20fps_h50_full_10k_train0000
```

训练集抽样结果：

```text
frame 0    mae_all26=0.000789  mae_arm14=0.001465  mae_hand12=0.000000
frame 66   mae_all26=0.002619  mae_arm14=0.004864  mae_hand12=0.000000
frame 132  mae_all26=0.004324  mae_arm14=0.008031  mae_hand12=0.000000
frame 199  mae_all26=0.007698  mae_arm14=0.014297  mae_hand12=0.000000
frame 265  mae_all26=0.005385  mae_arm14=0.010001  mae_hand12=0.000000
frame 332  mae_all26=0.000852  mae_arm14=0.001583  mae_hand12=0.000000
```

训练集误差显著低于留出集，说明模型对训练 episode 已经有明显过拟合效果；留出集动作趋势可用，但中间动作段误差更大，后续需要更多数据或更细的部署/控制调参验证。

### 7. 50 步 action chunk 轨迹图

- 为了观察预测的 50 步 action chunk 是否抖动，已为留出集和训练集各画 6 张轨迹图。每张图按维度分成：

```text
左臂 7
左手 6
右臂 7
右手 6
```

实线为模型预测，虚线为对应 episode 中未来 50 步真实 command。

- 留出集图：

```text
/data/caobochun/openpi/eval_outputs/tienkung_take_box_26d_20fps_h50_full_10k_holdout0013/trajectory_plots
```

- 训练集图：

```text
/data/caobochun/openpi/eval_outputs/tienkung_take_box_26d_20fps_h50_full_10k_train0000/trajectory_plots
```

- 图中横轴是模型 chunk step index `0..49`。当前采集和控制都按 20Hz 理解，因此完整 50 点 action chunk 约等于 2.5 秒；但当前部署代码还会根据 `policy_action_hz`、`control_hz`、`max_action_chunk_len`、`open_loop_horizon` 做插值、截断和重规划，因此真机实际执行时长以部署参数为准。

- 平滑度汇总文件：

```text
smoothness_summary.csv
smoothness_summary.png
```

留出集二阶差分最大值约 `0.006 ~ 0.030`，训练集二阶差分最大值约 `0.006 ~ 0.028`。从数值上没有特别大的尖峰，但图形上仍需要结合真实控制频率和插值策略判断是否足够平滑。

### 8. 当前结论和下一步

- 当前模型已经可以基于留出 episode 的视觉和状态输出 50 步动作 chunk。
- 训练集 `0000` 的误差明显较小，说明 1 万步全量 SFT 已经记住训练数据。
- 留出集 `0013` 中间动作段误差更大，说明 13 条训练数据对泛化仍偏少。
- 手部 12 维误差为 0，主要原因是当前数据里的手部 command 基本恒定，没有验证到复杂手部动作能力。
- 真机部署前需要重新检查部署侧配置，尤其是：

```text
control_hz=20
policy_action_hz=20
max_action_chunk_len
open_loop_horizon
request_when_remaining_steps
interpolation
```

### 9. 2026-06-15 频率口径修正

- 已重新确认当前采集频率和控制频率都按 20Hz 理解；之前日志中“真机部署控制循环高于采集频率”和“50 点约 1 秒”的说法是误解，已改回 20Hz 口径。
- 在 20Hz 口径下，模型一次输出的 50 步 action chunk 完整长度约为 2.5 秒。
- 训练好的模型配置和 checkpoint 不改：仍使用 pi0.5 默认 50 步预测、32 维模型 action/state，天工真实动作保持前 26 维。
- 部署侧源码默认值同步为 `control_hz=20`、`policy_action_hz=20`，状态/动作布局同步为 26 维。

这些参数决定 50 步 action chunk 在真机上被如何插值、截断和重规划，不能只按训练 fps 解释。

### 10. 2026-06-17 RTC 第一步：离线调度闭环（历史推理期尝试，已被 training-time RTC 替换）

> 说明：本节记录的是 2026-06-17 做过的 inference-time RTC / suffix soft blend / `rtc_guidance` 尝试。2026-06-18 之后项目主线已经改为 training-time RTC hard-prefix，部署侧统一传 `rtc_prefix`，不再使用本节里的 `rtc_guidance`、soft mask 和 suffix soft blend 作为主路径。保留本节是为了说明探索过程和为什么后续改方向。

- 已在服务器 `/data/caobochun` 使用 `proxychains4` 克隆官方 RTC 仓库：

```text
/data/caobochun/real-time-chunking-kinetix
commit 9296f31d62d5bfeb5779dcb2f9bcf71ca37f448b
```

该仓库是 Physical Intelligence 官方 Kinetix 仿真实验代码，不是 openpi 直接补丁；当前只借鉴异步执行、inference delay、execute horizon、prefix/suffix guidance 的算法结构。

- 已修正天工 RTC 部署侧调度：

```text
/data/caobochun/openpi/examples/tienkung/rtc_chunker.py
/data/caobochun/openpi/examples/tienkung/openpi_dual_hands_client.py
```

关键变化：`RtcChunker` 增加参数边界检查、基于剩余步数和延迟估计触发请求、忽略过期响应、返回动作 copy；RTC 控制器使用 `max_action_chunk_len` 作为可保留执行计划长度，用 `rtc_min_horizon` 控制最小重规划间隔，避免默认 `open_loop_horizon=8` 时没有旧 suffix 可融合。

- 已补充离线 smoke/replay 和单元测试：

```text
/data/caobochun/openpi/examples/tienkung/simulate_rtc_replay.py
/data/caobochun/openpi/examples/tienkung/rtc_chunker_test.py
```

验证命令：

```bash
cd /data/caobochun/openpi
. .venv/bin/activate
python -m py_compile examples/tienkung/rtc_chunker.py examples/tienkung/openpi_dual_hands_client.py examples/tienkung/simulate_rtc_replay.py src/openpi/policies/policy.py src/openpi/models_pytorch/pi0_pytorch.py
UV_LINK_MODE=copy uv run pytest examples/tienkung/rtc_chunker_test.py -q
UV_LINK_MODE=copy uv run python examples/tienkung/simulate_rtc_replay.py --num-chunks 6 --fixed-delay-steps 4 --output /data/caobochun/openpi/eval_outputs/rtc_replay_smoke.npz
```

结果：

```text
2 passed in 0.07s
rtc_replay_smoke.npz
num_actions=72
mean_delay_steps=4
max_delay_steps=4
mean_abs_delta=0.046723
max_abs_delta=0.151428
mean_abs_ddelta=0.015288
max_abs_ddelta=0.151428
```

- 已修正 policy server 的 RTC guidance shape：部署侧可传 26 维、32 步 guidance，`Policy._prepare_rtc_guidance()` 会经过输入 transform 后自动补齐/截断到模型 `action_horizon/action_dim`，避免开启 `rtc_model_guidance_enabled` 时和 pi0.5 默认 50x32 输出不匹配。

- 本次没有启动 ROS 客户端，没有连接真机，也没有发布机器人动作。下一步建议用已保存的真实 `rtc_action_chunk_*.npz` 或 holdout 评估输出跑 replay delay sweep，对比 async、固定 horizon RTC、delay-adaptive RTC、suffix soft blend 的平滑度和重规划频率。

### 11. 2026-06-18 RTC 方向修正：统一改为 training-time RTC hard-prefix

#### 为什么从 inference-time RTC 改到 training-time RTC

前一版 RTC 尝试里做过部署侧 suffix 软融合和模型侧 `rtc_guidance`。这条路线属于推理期 RTC：每次 denoising step 都要根据旧 chunk 的目标 prefix/postfix 计算修正项，核心会涉及 VJP/Jacobian 相关反传。对 pi0.5 这种大模型来说，这会明显增加 policy server 推理时间。

后续如果模型更大、图像更多、denoise step 更多，或者从 4090 换到 AGX 这类端侧设备，推理延迟 `d` 会继续变大。推理期 RTC 的 VJP/Jacobian 开销和延迟增长会互相放大：延迟越大，需要约束的 prefix 越长；prefix 越长，推理期修正越重。论文 `Training-Time Action Conditioning for Efficient Real-Time Chunking` 和 PI/Kinetix 参考项目都说明：在较大 delay 下，training-time RTC 通常比 inference-time RTC 更稳，而且部署时没有额外 VJP 开销。

因此本项目后续统一采用 training-time RTC：

```text
训练时：随机模拟推理延迟 d，把动作前 d 步作为无噪 action prefix 输入模型，只对 postfix 计算 flow matching loss。
推理时：把旧 chunk 中推理期间会继续执行的动作作为 rtc_prefix 传给模型，模型 hard-prefix denoise，返回后跳过已经执行过的 prefix，只执行 postfix。
```

旧的 `rtc_guidance`、soft mask、suffix soft blend 只作为历史尝试，不再作为主线。

#### H、s、d 和 25 的来源

当前 pi0.5 默认动作预测长度：

```text
H = action_horizon = 50
```

部署控制频率按 20Hz 计算：

```text
1 个控制步 = 1 / 20s = 50ms
```

当前 4090 上一次 policy 推理约 `73ms`：

```text
73ms / 50ms = 1.46 个控制步
```

因此 4090 上真实推理延迟按控制步计大约是 2 步以内，远小于 25。之前口头估算里出现过“约 15 个动作”的说法，容易和 denoise 内部计算/动作点数混在一起；训练 RTC 的 `d` 应按部署控制步换算。

AGX 上如果一次推理约 `1.2s`：

```text
1.2s / 50ms = 24 个控制步
```

RTC 合法条件是：

```text
d <= H - s
```

为了让训练覆盖 AGX 约 24 步延迟，同时保持一半 chunk 用于执行新 postfix，当前取：

```text
H = 50
s = 25
max d = 25
```

于是：

```text
H - s = 25
d ∈ [0, 25]
```

这既覆盖 4090 的小延迟，也覆盖 AGX 的约 24 步大延迟，并给后续推理开销继续变大的情况留出训练覆盖范围。

#### 当前训练实现

实现位置：

```text
/data/caobochun/openpi/src/openpi/models/pi0_config.py
/data/caobochun/openpi/src/openpi/models_pytorch/pi0_pytorch.py
/data/caobochun/openpi/src/openpi/models_pytorch/transformers_replace/models/gemma/modeling_gemma.py
/data/caobochun/openpi/src/openpi/training/config.py
/data/caobochun/openpi/scripts/train_pytorch.py
```

新增 `RTCTrainingConfig`，当前天工 RTC 配置：

```text
enabled=True
min_prefix_steps=0
max_prefix_steps=25
execution_horizon=25
prefix_probability=1.0
```

训练 forward 的核心逻辑：

```text
1. 每个 batch 样本随机采样 prefix_steps=d。
2. prefix_mask = arange(H) < d。
3. OpenPI flow 约定是 time=0 为清晰动作，time=1 为噪声。
4. prefix token 的 model_time 设为 0.0，x_t 直接等于真实动作。
5. postfix token 使用普通 sampled time，x_t = time * noise + (1 - time) * action。
6. pi0.5 action expert 的 adaRMS 条件改成支持 per-token timestep，即 [B, H, E]。
7. loss 只在 postfix 上计算，并按有效 postfix 步数归一化。
```

注意：论文伪代码使用的是相反 flow 记号，`time=1` 是清晰动作端；OpenPI 当前 PyTorch 实现里 `time=0` 是清晰动作端，所以代码中 prefix timestep 使用 `0.0` 是等价实现，不是反了。

训练配置：

```text
debug_pi05_rtc
pi05_tienkung_finetune_rtc
pi05_tienkung_dual_grippers_finetune_rtc
```

建议主训练命令：

```bash
cd /data/caobochun/openpi
. .venv/bin/activate
CUDA_VISIBLE_DEVICES=<gpu_id> python scripts/train_pytorch.py pi05_tienkung_finetune_rtc \
  --exp_name <实验名> \
  --overwrite \
  --batch-size <batch_size> \
  --num-workers <num_workers>
```

如果服务器没有 wandb key，可加：

```bash
WANDB_MODE=offline
```

#### 当前推理/部署实现

training-time RTC 部署侧只走 `rtc_prefix`，不再走 `rtc_guidance`：

```text
/data/caobochun/openpi/src/openpi/policies/policy.py
/data/caobochun/openpi/src/openpi/models_pytorch/pi0_pytorch.py
/data/caobochun/openpi/examples/tienkung/rtc_chunker.py
/data/caobochun/openpi/examples/tienkung/openpi_dual_hands_client.py
```

部署流程：

```text
1. RtcChunker 根据当前 chunk 剩余动作和延迟 buffer 估计 d。
2. 请求下一次推理前，把旧 chunk 剩余动作补齐成 action_prefix，并只声明前 d 步有效。
3. observation 中加入：
   rtc_prefix = {
       "action_prefix": action_prefix,
       "delay": d,
   }
4. Policy._prepare_rtc_prefix() 让 prefix action 经过同一套 action transform，转换到模型采样空间。
5. PI0Pytorch.sample_actions() 每个 denoise step 都 hard overwrite prefix token，并给 prefix token 使用 clean endpoint timestep。
6. 新 chunk 返回后，部署侧按真实 observed_delay 跳过已经执行过的 prefix，只执行 postfix。
```

已经从 `examples/tienkung` 删除旧推理期 RTC 配置和接口：

```text
rtc_model_guidance_enabled
rtc_guidance_beta
rtc_guidance_decay
rtc_guidance_eps
rtc_blend_steps
rtc_soft_preserve_weight
rtc_guidance
soft mask
suffix soft blend
```

#### 已完成验证

已完成的最小验证：

```text
1. PyTorch 真实 batch forward/backward：loss finite。
2. hard-prefix inference sample：sample_shape=(1, 50, 32)，sample_finite=True。
3. train_pytorch 2 step smoke：pi05_tienkung_finetune_rtc 可正常训练和保存 checkpoint。
4. examples/tienkung 编译通过。
5. examples/tienkung/rtc_chunker_test.py：2 passed。
6. examples/tienkung/simulate_rtc_replay.py hard-prefix smoke 通过。
```

本次没有启动 ROS 客户端，没有连接真机，也没有发布机器人动作。

#### 后续改进点

1. 增加 delay 分布配置：

```text
uniform      # 当前 [0, 25] 均匀采样
exp          # 借鉴 Kinetix，偏向小 delay，但仍覆盖大 delay
empirical    # 用真机/AGX 实测延迟直方图采样
```

2. 做离线 delay sweep：

```text
d = 0..25
s = 25
num_steps = 10
```

对比普通 async、training-time RTC hard-prefix、历史 inference-time guidance 的动作连续性、重规划频率和任务成功率。

3. 训练策略建议从普通 BC checkpoint 继续 fine-tune RTC，而不是训练一个新的输出头。Kinetix 官方复现也是从普通 flow policy checkpoint 加 simulated delay 继续训练。

4. 如果以后使用 JAX `scripts/train.py`，需要把 PyTorch 这套 training-time RTC 同步到 JAX `src/openpi/models/pi0.py`。当前完整实现和验证只覆盖 PyTorch pi0.5 / `scripts/train_pytorch.py` 路径。

5. 当前 `.venv` 中 transformers 的 Gemma replacement 已 patch 支持 per-token adaRMS；如果重建虚拟环境，需要重新复制：

```bash
cp -r ./src/openpi/models_pytorch/transformers_replace/* .venv/lib/python3.11/site-packages/transformers/
```

6. 真机部署前检查：

```text
policy server 是否加载 RTC finetune checkpoint
client 是否使用 controller_mode="rtc"
control_hz=20
policy_action_hz=20
max_action_chunk_len 是否和模型输出/部署截断一致
rtc_min_horizon / rtc_initial_delay_steps 是否符合当前设备实测延迟
急停、限幅、动作维度、三路图像话题是否正确
```

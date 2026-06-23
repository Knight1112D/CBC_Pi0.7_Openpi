# Training-Time RTC 设计和实现说明

## 背景

天工部署需要异步推理：控制循环保持 20Hz 发布动作，policy server 在后台生成下一段 action chunk。推理期间机器人不能停住，只能继续执行上一段 chunk 已经承诺的动作。这些在新 chunk 前面重叠的动作就是 RTC 的 action prefix。

早期尝试过 inference-time RTC：部署侧构造旧 chunk 目标和 soft mask，模型采样时通过 VJP/Jacobian 相关反传修正速度场。这个方法灵活，但对 pi0.5 这类大模型很贵，denoise 每一步都可能增加额外反传开销。随着模型更大、设备更慢或 denoise step 增多，推理延迟 `d` 会继续变大，VJP 开销会进一步放大。

因此当前改为 training-time RTC：训练时模拟推理延迟，把 prefix 作为无噪动作条件输入模型；部署时只传 hard-prefix，不再做推理期 VJP 修正。

## 参数选择

当前 pi0.5 默认：

```text
H = action_horizon = 50
```

控制频率：

```text
control_hz = 20
1 个控制步 = 50ms
```

实测和预估：

```text
4090 推理一次约 73ms，约 73 / 50 = 1.46 个控制步
AGX 推理一次如果约 1.2s，约 1200 / 50 = 24 个控制步
```

RTC 合法条件：

```text
d <= H - s
```

为了覆盖 AGX 约 24 步延迟，同时保留足够 postfix 给新策略执行，当前选择：

```text
H = 50
s = 25
max d = 25
d 随机采样范围 = [0, 25]
```

4090 的真实延迟远小于 25，但训练覆盖到 25 可以兼容 AGX 和后续更大推理开销。

## 训练实现

核心文件：

```text
src/openpi/models/pi0_config.py
src/openpi/models_pytorch/pi0_pytorch.py
src/openpi/models_pytorch/transformers_replace/models/gemma/modeling_gemma.py
src/openpi/training/config.py
scripts/train_pytorch.py
```

新增配置：

```text
RTCTrainingConfig(
    enabled=True,
    min_prefix_steps=0,
    max_prefix_steps=25,
    execution_horizon=25,
    prefix_probability=1.0,
)
```

训练逻辑：

```text
1. 每个 batch 样本随机采样 d。
2. prefix_mask = arange(H) < d。
3. prefix token 输入真实无噪动作。
4. postfix token 使用普通 flow matching 加噪。
5. prefix token 的 timestep 使用 OpenPI flow 约定下的 clean endpoint：time=0.0。
6. postfix token 使用采样 timestep。
7. loss 只计算 postfix，并按有效 postfix 步数归一化。
```

注意：论文伪代码中 clean endpoint 写作 `time=1.0`，OpenPI 当前 PyTorch flow 公式是：

```text
x_t = time * noise + (1 - time) * action
```

所以 OpenPI 中 clean endpoint 是 `time=0.0`。

## 推理和部署实现

核心文件：

```text
src/openpi/policies/policy.py
src/openpi/models_pytorch/pi0_pytorch.py
examples/tienkung/rtc/rtc_chunker.py
examples/tienkung/deploy/openpi_dual_hands_client.py
```

部署侧传入：

```python
observation["rtc_prefix"] = {
    "action_prefix": action_prefix,
    "delay": delay,
}
```

模型采样时：

```text
1. 每个 denoise step 都把 prefix 区间 hard overwrite 成 action_prefix。
2. prefix token timestep 固定为 time=0.0。
3. postfix token timestep 使用当前 denoise time。
4. 返回完整 chunk 后，部署侧按真实 observed_delay 跳过已经执行过的 prefix，只执行 postfix。
```

`examples/tienkung` 已经移除旧推理期 RTC 字段：

```text
rtc_guidance
rtc_model_guidance_enabled
rtc_guidance_beta
rtc_guidance_decay
rtc_guidance_eps
rtc_blend_steps
rtc_soft_preserve_weight
```

## 部署调度和计时

普通 async 控制器现在默认：

```text
request_immediately_after_chunk=True
```

也就是收到上一段 action chunk、填充执行队列后，如果没有正在进行的推理，会立刻用最新观测提交下一次异步推理。`request_when_remaining_steps` 保留为关闭立即重规划后的低水位触发参数。

RTC 控制器仍然使用 `RtcChunker.should_request()`，不会每拿到 chunk 就立刻重规划。它需要等执行步数达到 `rtc_min_horizon` 或剩余动作不足以覆盖延迟估计时再请求，避免破坏 `s/d/prefix/postfix` 的对齐关系。

推理返回会携带三层计时：

```text
client_timing.websocket_infer_ms       # 客户端 websocket infer 调用总耗时
server_timing.recv_ms                  # server 等待/接收请求耗时
server_timing.unpack_ms                # msgpack 解包耗时
server_timing.infer_ms                 # server 调 policy.infer 耗时
server_timing.pack_ms                  # msgpack 打包响应耗时
policy_timing.observation_tokenize_ms  # OpenPI input transform/tokenize 耗时
policy_timing.rtc_prefix_transform_ms  # rtc_prefix 动作进入模型空间的 transform 耗时
policy_timing.tensorize_ms             # numpy 到 JAX/PyTorch batch tensor 耗时
policy_timing.model_sample_ms          # model.sample_actions 耗时
policy_timing.output_transform_ms      # 模型输出转回机器人动作空间耗时
policy_timing.action_ready_ms          # policy 收到 observation 到动作可返回的总耗时
model_timing.vlm_prefix_forward_ms     # 图像/语言 prefix KV 的 VLM 前向耗时
model_timing.flow_denoise_ms           # flow matching 迭代降噪耗时
model_timing.flow_denoise_steps        # 实际 denoise step 数
```

CUDA 计时在模型阶段边界做同步，因此数值比纯 Python wall-clock 更接近真实 GPU 执行时间，但会引入少量测量开销。部署客户端日志会打印紧凑摘要，保存 action chunk 时也会把 timing 字典写入 `.npz`。

相机话题和消息类型已按 `Knight1112D/Tienkung_vla_collect_data` 的 `main@7899e361b32826b3ea3020ef2cc31cdb7a9c779a` 对齐：

```text
头部: /camera/color/image_raw/compressed, sensor_msgs/msg/CompressedImage
左手: /camera/d405_left/color/image_h264, foxglove_msgs/msg/CompressedVideo
右手: /camera/d405_right/color/image_h264, foxglove_msgs/msg/CompressedVideo
```

## 训练命令

单卡 smoke：

```bash
cd /data/caobochun/openpi
. .venv/bin/activate
CUDA_VISIBLE_DEVICES=5 python scripts/train_pytorch.py pi05_tienkung_finetune_rtc \
  --exp_name rtc_smoke \
  --overwrite \
  --num-train-steps 2 \
  --batch-size 1 \
  --num-workers 0 \
  --save-interval 2 \
  --no-wandb-enabled
```

正式训练示例：

```bash
cd /data/caobochun/openpi
. .venv/bin/activate
CUDA_VISIBLE_DEVICES=5,7 WANDB_MODE=offline torchrun --standalone --nnodes=1 --nproc_per_node=2 \
  scripts/train_pytorch.py pi05_tienkung_finetune_rtc \
  --exp_name tienkung_take_box_26d_rtc_h50_s25_d25 \
  --overwrite
```

## 已验证内容

```text
真实 batch forward/backward 通过，loss finite。
hard-prefix sample_actions 通过，输出 shape=(1, 50, 32)。
train_pytorch 2 step smoke 通过。
examples/tienkung 编译通过。
examples/tienkung/rtc/rtc_chunker_test.py 通过。
examples/tienkung/eval/simulate_rtc_replay.py hard-prefix smoke 通过。
```

## 后续建议

1. 增加 delay 分布配置：`uniform`、`exp`、`empirical`。
2. 用真实 4090/AGX 延迟日志生成 empirical delay histogram。
3. 做离线 delay sweep：`d=0..25, s=25`。
4. 对比普通 async、training-time RTC hard-prefix、历史 inference-time guidance。
5. 如果后续使用 JAX 训练，需要同步实现到 `src/openpi/models/pi0.py`。
6. 重建 `.venv` 后重新复制 transformers replacement，确保 Gemma adaRMS 支持 per-token 条件。

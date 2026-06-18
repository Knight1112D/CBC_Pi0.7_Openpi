# CBC OpenPI 天工项目说明

这个目录记录 CBC 在 OpenPI 上做天工机器人训练、部署和 RTC 改造的项目级说明。原始 OpenPI 文档仍保留在仓库根目录和 `docs/` 其他文件中；这里的内容只描述本项目新增的天工数据、pi0.5 训练、异步部署和 training-time RTC。

## 文档索引

- `training_time_rtc.md`：training-time RTC 的设计原因、H/s/d 参数、训练实现、部署 hard-prefix 流程和后续实验建议。
- `/data/caobochun/openpi/cbc开发日志.md`：按时间记录的开发日志，包含数据转换、训练、部署和 RTC 改造过程。

## 当前主线

当前项目主线是：

```text
pi0.5 PyTorch checkpoint
-> 天工 LeRobot 数据训练
-> training-time RTC fine-tune
-> examples/tienkung 使用 rtc_prefix hard-prefix 异步部署
```

推理期 `rtc_guidance` / VJP / soft mask 是历史尝试，不再作为天工 RTC 主路径。后续实验应优先使用 training-time RTC checkpoint，并在部署侧传 `rtc_prefix`。

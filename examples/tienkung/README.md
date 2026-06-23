# 天工 OpenPI 示例目录

这个目录按用途拆分天工 VLA 的数据转换、离线评估、真机部署和 RTC 工具。根目录只保留目录说明和包初始化文件，真实实现都放在子目录中。

## 目录结构

```text
examples/tienkung/
  common/   # 部署、评估和回放共用代码
  configs/  # 部署 JSON 示例配置
  data/     # 原始数据到 LeRobot 的转换脚本
  deploy/   # ROS2 真机部署客户端
  eval/     # 离线评估、RTC replay 和 smoke 脚本
  rtc/      # RTC chunk 调度器和单元测试
  scripts/  # 训练或运维 shell 入口
```

## 常用入口

```bash
python examples/tienkung/deploy/openpi_dual_hands_client.py --config-path examples/tienkung/configs/tienkung_dual_hands_config.example.json
python examples/tienkung/eval/simulate_rtc_replay.py
python examples/tienkung/eval/eval_tienkung_holdout.py
pytest examples/tienkung/rtc/rtc_chunker_test.py -q
```

## 安全说明

`deploy/openpi_dual_hands_client.py` 会连接 ROS2 并发布机器人动作。真机部署前需要确认 policy server、动作维度、关节顺序、限幅、控制频率、急停和相机话题。

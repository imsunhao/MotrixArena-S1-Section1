# Section1 修订训练路线

## 决策

当前路线固定为同学已验证的环境结构：78 维观测、24 点地形扫描、四拍参考步态、
PPO residual、NumPy/MotrixSim 环境和 Torch/skrl PPO。只参考环境和课程结构，所有
checkpoint 均由本项目重新训练，不加载同学权重。

固定条件课程已经完成：

| 阶段 | 环境 | 训练步数 |
| --- | --- | ---: |
| S1 | `vbot-section01-s1-velocity-course` | 9,600 |
| S2 | `vbot-section01-s2-terrain-course` | 16,000 |
| S3a | `vbot-section01-s3a-uphill-course` | 16,000 |
| full | `vbot-section01-full-course-v2-train` | 6,400 |

本项目固定起点权威 checkpoint 为
`artifacts/checkpoints/course_torch/fixed_full_seed45_agent3600.pt`。它从
`x=0, y=-2.4, yaw=+Y` 连续评估 5 回合全部到达平台，未使用同学 checkpoint。

## 新增难度顺序

后续严格按以下顺序推进，前一级未通过时不启动下一级：

1. random XY：`x in [-0.5,0.5]`、`y in [-2.9,-2.0]`、固定 yaw、无停稳要求；
2. yaw：在通过 random XY 的配置上只增加 `+-0.15 rad` 初始航向扰动；
3. stable：在通过 yaw 的配置上只增加平台连续稳定 50 个控制步。

random XY 不再只微调 full。它重新执行同样的四阶段长度，让基础直行、坑洼和坡面
课程都覆盖位置扰动：

| 阶段 | 环境 | 训练步数 | 相对固定课程的唯一变化 |
| --- | --- | ---: | --- |
| S1 | `vbot-section01-peer-xy-s1-course` | 9,600 | 正式起点 random XY |
| S2 | `vbot-section01-peer-xy-s2-course` | 16,000 | 正式起点 random XY |
| S3a | `vbot-section01-peer-xy-s3a-course` | 16,000 | 坡前局部起点 XY 抖动 |
| full | `vbot-section01-peer-xy-full-course` | 6,400 | 正式起点 random XY |

四个环境的 `initial_yaw_noise=0`、`stable_hold_seconds=0`，控制器、观测、奖励和 PPO
参数继续沿用固定课程。训练入口会在未显式传 `--stage-steps` 时使用表中的标准长度。

## 验收

每阶段先从中间 checkpoint 扫描候选，再做多个独立单环境 seed 评估。MotrixSim 在
`num_envs>1` 与单环境下存在数值轨迹差异，因此最终证据只使用 `num_envs=1`。

random XY 晋级要求：正式 XY 分布的多个 seed 均出现可重复平台到达，同时固定起点
回归不丢失。yaw 和 stable 阶段仍分别使用相同的正式出生分布评估，最终成功必须同时满足
`ever_on_platform_rate>0` 与 `stable_success_rate>0`。

## 最终验收结果

用户最终将停稳门从 100 个控制步调整为 50 个控制步。为保留实验口径，原
`vbot-section01-peer-xy-yaw-stable-v4-course` 继续表示 100 步严格门；新增
`vbot-section01-peer-xy-yaw-stable-v4-50-course`，只把
`stable_hold_seconds` 从 1.0 改为 0.5，其他出生分布、yaw、控制、奖励和平台判定完全相同。

最终 checkpoint 为
`artifacts/checkpoints/course_torch/peer_xy_full_seed345_agent5200.pt`，由本项目训练，
SHA-256 为
`b6f667f194b50905dd5451b128838ee56b3ad1acf20460f74f4881928c460646`。

逐级验收：

| 难度 | 正式回合 | 上平台 | 稳定成功 |
| --- | ---: | ---: | ---: |
| fixed | 8 | 8/8 | 不要求 |
| random XY | 64 | 17/64 | 不要求 |
| random XY + `+-0.15 rad` yaw | 64 | 18/64 | 不要求 |
| random XY + yaw + stable-50 | 128 | 28/128 | 3/128 |

stable-50 成功来自三个不同评估 seed：2026、2090、2122。最终视频使用 seed2026 的
第 8 回合，该回合实际达到 100 个连续稳定步，因此无需因门槛改为 50 步而重新录制。

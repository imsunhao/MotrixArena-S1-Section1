# Section1 最终训练课程

## 设计原则

最终路线使用 NumPy/MotrixSim 环境与 Torch/skrl PPO。观测为 78 维，其中包含 24 点地形高度扫描；控制由四拍参考步态与 PPO residual 共同生成。所有训练权重均由本项目重新训练。

训练按固定起点、随机 X/Y、航向扰动和平台停稳逐级增加难度。每次只增加一个变量，并从已经掌握当前地形阶段的 checkpoint 继续训练，避免位置分布变化导致地形技能遗忘。

## 环境阶段

| 阶段 | 环境 | 默认步数 | 作用 |
| --- | --- | ---: | --- |
| S1 | `vbot-section01-s1-velocity-course` | 9,600 | 平地直行与朝向 |
| S2 | `vbot-section01-s2-terrain-course` | 16,000 | 坑洼区域通过能力 |
| S3a | `vbot-section01-s3a-uphill-course` | 16,000 | 坡面持续上行 |
| full | `vbot-section01-full-course-v2-train` | 6,400 | 固定起点完整路线 |
| XY-S1 | `vbot-section01-xy-s1-course` | 9,600 | 正式随机 X/Y 平地段 |
| XY-S2 | `vbot-section01-xy-s2-course` | 16,000 | 正式随机 X/Y 坑洼段 |
| XY-S3a | `vbot-section01-xy-s3a-course` | 16,000 | 坡前随机位置 |
| XY-full | `vbot-section01-xy-full-course` | 6,400 | 正式随机 X/Y 完整路线 |
| stable-50 | `vbot-section01-xy-yaw-stable-v4-50-course` | 6,400 | 航向扰动与平台停稳 |

## 稳定判定

控制周期 `ctrl_dt=0.01 s`，最终稳定门为 `stable_hold_seconds=0.5 s`，即连续 50 个控制步。判定同时要求机器人位于平台范围内、保持直立，并满足线速度、竖直速度和角速度阈值。

保留 100 步环境作为更严格的控制实验，但最终验收以 50 步为准。最终成果视频中的成功回合实际连续稳定 100 步，因此同时满足两个口径。

## 权威结果

最终 checkpoint 为：

`artifacts/checkpoints/course_torch/section1_xy_full_seed345_agent5200.pt`

SHA-256：

`b6f667f194b50905dd5451b128838ee56b3ad1acf20460f74f4881928c460646`

正式 random XY + yaw + stable-50 评估共 128 回合，28 回合曾踏上平台，3 回合达到连续稳定 50 步，成功覆盖 `2026`、`2090`、`2122` 三个独立评估 seed。

最终交付保留三个独立完整视频，不做拼接。三个视频分别对应上述三个成功 seed，起点 X 覆盖负侧、中间正侧和接近正边界的位置，能够直接展示正式随机 X/Y 出生分布下的完整路线结果。

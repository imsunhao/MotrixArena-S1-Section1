# MotrixArena S1 Section 1

本项目训练 VBot 四足机器人从 Section 1 起跑线出发，穿越坑洼区和 15° 上坡，最终踏上 2026 平台并连续停稳。

## 成功口径

- 辅助指标 `ever_on_platform_rate`：回合中曾经踏上平台。
- 主指标 `stable_success_rate`：机器人位于平台区域、保持直立和低速状态至少 1 秒。

精确停在平台中心不是本项目的硬性成功条件。

## 官方资源

仓库不包含约 533 MB 的 MotrixArena 官方资源、训练 checkpoint 和视频。运行前执行：

```bash
./scripts/setup_arena_section1_assets.sh
```

资源来源为 MotrixLab `MotrixArena-S1` 分支 README 公布的 starter kit。

## 当前进度

- [x] 建立独立 Arena S1 环境
- [x] 修正起跑线出生范围、目标点和初始朝向
- [x] 实现“曾踏上平台”和“平台停稳”双成功指标
- [ ] 完成第一版导航与越障奖励
- [ ] 小规模训练验证
- [ ] 完整训练、统一评估与视频录制

# ADMET Competition Model — 当前交接 v366

更新于 2026-11-04（香港时间）。用户反馈的官方最好成绩为 **v364：51.269**；最新 **v366 是尚未取得官方成绩的实验候选**。目标 52 尚未达到。

先读 [当前交接说明](HANDOFF_CURRENT.md)。完整源码在 `admet-v366-source.zip`，固定输入、两阶段证据、六份最终权重及 CSV 在 `admet-v366-runtime.tar.gz.partNNN`，逐文件校验清单为 `handoff-manifest-v366.json`。

```bash
python restore_v366_handoff.py --bundle-dir . --destination ../ADMET-v366-restored
```

|版本|状态|本轮改动|CSV SHA256|
|---|---|---|---|
|v364|官方 51.269，当前最好|脂溶性、溶解度各加入 20% 相关任务迁移图 Transformer|`777eca282b59101463fe02f71802c24502a87ba7352482b24a64943ef81ede79`|
|v366|实验候选，官方未知|在 v364 上仅更新 LD50，20% 图 Transformer 迁移专家；其余 21 项保留|`45982f798555d88f2c934d187c709091db02841e257c1f2450b470b9a02c04fc`|

v366 的 LD50 四分区 MAE 相对下降 0.708%、0.822%、1.475%、1.073%；骨架 bootstrap 有一区间跨零，未通过严格稳定性验收。PPBR 确认失败，未采用。部分历史集成 OOF 仍为代理，不能保证隐藏测试提分。

GPU 训练只能在用户指定的远端执行。上传内容不含 SSH 密码或访问令牌。当前状态以本节、`HANDOFF_CURRENT.md`、`CURRENT_STATUS.json` 为准；下文保留旧快照，不描述当前进展。

## 历史首页快照：v145

当前有官方成绩证据的最佳版本是 **v145：51.166**，目标为 52。最高分版本的 22 项成绩、代码入口、预测来源、校验记录与后续实验结论见 [成果总览](docs/V145_RESULTS_AND_CODE_GUIDE.md)。

- [v145 官方完整成绩单](docs/platform_scorecard_v145_20261027.json)
- [v145 路由修复说明](docs/ITERATION_V145_20261027.md) · [CSV 交付校验](docs/v145_local_delivery.json)
- [v145 代码](tools/v145_feedback_repair.py) · [固定基线重建](tools/v146_current_baseline.py) · [冻结验证门槛](tools/v148_experiment_runner.py)
- [最近已核验结果 v290](docs/v290_final_status.json) · [v294 可行性审查](docs/v294_context_uncertainty_audit.json) · [v295 推理优化实验代码](tools/v295_v145_context_integration.py)
- [GitHub 内容范围](docs/GITHUB_UPLOAD_CONTENTS.md) · [状态快照](CURRENT_STATUS.json)

v145 是 v130 与 v144 已有输出的整任务路由修复：保留 v144 的 AMES、CYP2C9 抑制和 CYP2D6 抑制，其余 19 项采用 v130。相对 v130 共改变 6,502 条预测。它不是单个新训练模型，完整复现需要历史输出、数据与权重。

此快照没有经核验优于 v145 的新 CSV。v295 代码已整理，但未核验其远端最终结果，不能当作已完成或已提分版本。内部验证不能保证隐藏测试涨分。当前按用户要求只交付通过原固定门槛的 CSV，由用户手动提交并提供成绩单；不自动上传比赛平台或读取新成绩。

此次更新以源码和证据为主：不包含新数据集、模型权重、特征数组、release 归档、运行环境或 SSH 凭据；v145 CSV 由本地路径和 SHA256 标识，未纳入本次上传。历史仓库及旧 Release 的内容保持原样。

[历史 README 快照](docs/HISTORY_THROUGH_V169.md) · [早期恢复说明](docs/REPRODUCE.md) · [早期交接日志](docs/HANDOFF.md)

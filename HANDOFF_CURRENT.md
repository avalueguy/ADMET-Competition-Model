# 当前模型与诊断交接

截至2026-11-06（香港时间），模型最佳用户报告成绩：v387=51.314。v389=51.313，回退v387；v390完整实验无采用任务。目标52分和平均MAE降低5%尚未实现。

1. 读取 `docs/V387_RESULTS_AND_HANDOFF.md`、`docs/V389_OFFICIAL_REGRESSION_AND_V390_PLAN.md`、`docs/V390_RESULTS_AND_HANDOFF.md`。
2. v387模型CSV为 `v387-official-best.csv`，SHA256 `97cdbd749fde3ea10c2e55cfea143cfc2e0208b18edc09ebd101feb3d72a77a3`。完整成绩是用户归属，平台未提供上传文件字节哈希。
3. 从[原交接Release](https://github.com/avalueguy/ADMET-Competition-Model/releases/tag/handoff-v389-20261106)下载source、manifest、restore脚本及五份运行归档，按原manifest核对哈希；执行 `python restore_v389_handoff.py --bundle-dir . --destination ../ADMET-restored`。顺序为v380→v382→v385→v388→v389。
4. 将[当前源码快照](archives/research-sync-20261106/README.md)解压到另一个目录核对，再覆盖恢复目录中的源码与文档。快照不会声称覆盖所有早期缺失权重；未重训强历史集成，部分OOF仍是proxy。
5. 若复核v390，从[本轮Release](https://github.com/avalueguy/ADMET-Competition-Model/releases/tag/research-sync-20261106-offline-diagnostics)取 `v390_complete_results.tar.gz`，按 `docs/v390_archive_verification.json` 核对。120个不可变输出通过，verification.log追加PASS的活动日志例外已有记录。
6. 后续GPU拟合仅使用用户指定 `connect.nmb2.seetacloud.com:25870`；凭据通过运行时安全输入，不写入仓库。优先恢复历史组件的真实骨架OOF，再验证Caco2条件信息和新表示。

`diagnostics/tdc_public_test_overlap/`与训练分离。这里包含公开TDC测试观测、完全标签查表CSV（平台拒绝）和61.775810247分合成CSV（本地评分、未上传）。用户称“标准答案”的文件是外部公开观测，不是合规模型输出，不能加入比赛训练或提交。

公开标签复算v387时21项一致；溶解度本地MAE0.7008209745143054、用户上报0.7006168876998713，差异尚未解决。保留原CSV字节，不通过改动标签或噪声伪造模型涨分。

最新状态看 `CURRENT_STATUS.json`，历史首页已保存至 `docs/history/`，历史实验报告保持原有时间与结论。

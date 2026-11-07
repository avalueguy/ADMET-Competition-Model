# ADMET 当前交接：v387 官方 51.314 / v389 实验

截至 2026-11-06，官方最好是 **v387：51.314**。v389 尚未取得官方成绩，仅修改 PPBR，四个本地分区 MAE 降低0.13%–0.66%，确认置信区间跨零。平均 MAE 降低5%的目标尚未达到。

下载 [本轮完整交接 Release](https://github.com/avalueguy/ADMET-Competition-Model/releases/tag/handoff-v389-20261106) 的全部资产，或使用根目录的源码包和两个 CSV。**GitHub 自动生成的 Source code 归档只对应发布时主分支，不能代替本轮手工上传的源码和模型资产。**

1. 安装 Python 3.10+；恢复脚本只用标准库。
2. 下载 `admet-v389-source.zip`、`handoff-manifest-v389.json`、`restore_v389_handoff.py` 和5份手工上传的 `.tar.gz`，放在同一目录。
3. 执行 `python restore_v389_handoff.py --bundle-dir . --destination ../ADMET-v389-restored`。
4. 在恢复目录先读 `docs/V389_RESULTS_AND_HANDOFF.md`、`docs/v389_delivery.json` 和 `docs/v388_dependency_audit.json`，按原脚本与锁定依赖在指定远端继续训练。SSH 密码不在交接内容中。

模型和输入按 v380 → v382 → v385 → v388 → v389 恢复；源码包最后覆盖为当前源码。恢复程序先核验压缩包 SHA256，再检查解包路径，拒绝链接和路径越界。

当前源码、训练脚本、历史书面证据在 `admet-v389-source.zip`；模型、固定输入、预测和分区证据在5份运行归档。历史集成部分 OOF 是代理、早期模型不完整等限制继续保留，不能声称对任意新分子完整重建整个历史集成。

|提交|状态|SHA256|
|---|---|---|
|`v387-official-best.csv`|用户反馈官方51.314|`97cdbd749fde3ea10c2e55cfea143cfc2e0208b18edc09ebd101feb3d72a77a3`|
|`v389-experimental.csv`|尚无官方成绩|`c02dff23296dd35458ac1e4fb1ceb094bfdc686a7bde5c3de6aed81cf76be20c`|

CSV 在比赛上传时应按平台要求恢复原文件名，见清单中的 `original_path`。不自动提交比赛。

以下为本轮实验结果；更早版本的说明属于历史记录。


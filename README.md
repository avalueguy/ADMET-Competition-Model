# ADMET 模型研究与交接

更新于2026-11-06（香港时间）。模型最佳用户报告成绩为 **v387：51.314**。最新完成v390没有通过采用门槛，52分及平均MAE降低5%的目标尚未达到。

|材料|成绩或结论|入口|
|---|---|---|
|v387模型预测|用户报告51.314，上传字节未由平台核对|[模型CSV](v387-official-best.csv) · [源码快照](archives/research-sync-20261106/) · [完整交接](HANDOFF_CURRENT.md)|
|v389组合实验|用户报告51.313，已退回v387|[结果说明](docs/V389_OFFICIAL_REGRESSION_AND_V390_PLAN.md)|
|v390辅助残差实验|64次远端GPU种子拟合，无采用任务|[实验结果](docs/V390_RESULTS_AND_HANDOFF.md) · [训练代码](tools/v390_auxiliary_residual.py)|
|61分离线合成文件|本地61.775810247；公开标签加随机扰动，非模型成绩|[合成CSV](diagnostics/tdc_public_test_overlap/synthetic_offline/OFFLINE_SYNTHETIC_seed20261106.csv) · [逐项得分](diagnostics/tdc_public_test_overlap/synthetic_offline/synthetic_score_report.json)|
|公开“标准答案”|TDC公开测试观测查表，纯标签文件已被平台拒绝|[公开标签CSV](diagnostics/tdc_public_test_overlap/public_labels/PUBLIC_TEST_LABELS_NOT_MODEL.csv) · [来源与限制](diagnostics/tdc_public_test_overlap/)|

61.776分是有已知标签参与构造的离线评分测试，不代表模型提升。赛事规则禁止外部数据并要求模型生成预测，这些诊断文件不能作为合规模型提交。模型基线与诊断材料分开保存。

从[当前交接](HANDOFF_CURRENT.md)开始更新模型；查找源码、历史记录和归档请读[仓库目录](docs/REPOSITORY_INDEX.md)。本轮同步范围、校验方式和未打包项见[上传内容说明](docs/GITHUB_UPLOAD_CONTENTS.md)。

本轮所有最新源码、书面证据和轻量实验记录另有[完整源码快照](archives/research-sync-20261106/README.md)。v390完整运行产物在[本轮Release](https://github.com/avalueguy/ADMET-Competition-Model/releases/tag/research-sync-20261106-offline-diagnostics)；v380至v389的递归输入与模型包保留在[原交接Release](https://github.com/avalueguy/ADMET-Competition-Model/releases/tag/handoff-v389-20261106)。

保留历史源码的原有路径，避免破坏import、相对路径和冻结签名。训练依赖与旧完整模型限制见交接说明。SSH密码、令牌、本地运行环境与node_modules不纳入同步。

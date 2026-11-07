# 仓库目录

|位置|内容与使用方法|
|---|---|
|README.md / HANDOFF_CURRENT.md|当前状态和接手步骤，优先阅读|
|CURRENT_STATUS.json|当前模型状态；历史releases条目是当时快照|
|tools/|版本化训练、验证、交付脚本，原路径保留|
|docs/|计划、成绩单、失败结果、哈希、交付证据|
|docs/history/|整理前首页与状态，保留历史表述|
|diagnostics/tdc_public_test_overlap/|外部公开标签与61.776分合成数据，严格分开标注|
|archives/research-sync-20261106/|当前全部源码和轻量证据快照，附逐文件manifest|
|源码快照内github_v387_code/|此前发布的v387冻结代码包，原始内容与哈希不改|
|v387-official-best.csv|模型最佳用户报告51.314的原始预测文件|
|v389-experimental.csv|已被后续反馈否定的历史实验，51.313|
|experiments/|本地固定输入、权重和运行证据，体积较大；源码快照含轻量证据，完整数据在Release|
|根目录admet_v*.py|历史训练入口；仍被旧脚本引用，因此不随意移动|

本轮Release提供v390完整运行归档、源码快照、诊断打包与校验清单。以前的Release和根目录v366分卷、v389包保留，避免旧链接失效。旧源码与数据的完整复现限制见当前交接说明。

目录整理由统一入口、用途分类、来源说明和manifest组成，不改写历史实验的冻结脚本或签名。node_modules、虚拟环境、缓存、临时探测响应、浏览器截图及凭据不上传。

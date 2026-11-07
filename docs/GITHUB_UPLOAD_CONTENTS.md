# 2026-11-06 GitHub同步内容

目标仓库：https://github.com/avalueguy/ADMET-Competition-Model。本次由用户明确要求同步最新内容，包括公开标签及61分离线合成文件。

- 主分支入口更新：README、当前交接、目录索引、状态文件、v390结果与最新训练/验证代码。
- 当前源码快照：根目录历史源码、tools和docs的最新文本、v387已发布包、experiments的轻量JSON/Python记录。保留原路径、逐文件SHA256；不把历史快照当作新的实验结论。
- 公开标签隔离目录：原始公开TDC ZIP、标签CSV、matched_rows、比赛模板、得分参考、逐行重合和平台拒绝证据。
- 合成诊断：固定种子20261106，4,917行扰动，本地61.775810247，原CSV字节与报告完整保留。非模型成绩、非合规参赛文件、没有平台成绩。
- v390完整运行产物：`v390_complete_results.tar.gz`，SHA256 `f765c95280a61dc408592860e7ef06b3aa03807cafddff5b5dd0a3ce2e2c1f80`，68,706,342字节，在[本轮Release](https://github.com/avalueguy/ADMET-Competition-Model/releases/tag/research-sync-20261106-offline-diagnostics)。
- v380→v389所需输入/模型包继续从[现有Release](https://github.com/avalueguy/ADMET-Competition-Model/releases/tag/handoff-v389-20261106)获取，不重复改名或改写。

不上传：SSH密码、访问令牌、Cookie、虚拟环境、node_modules、浏览器截图、临时下载探测副本及未独立收集的远端候选权重。不移动被模型脚本引用的历史源码。大文件通过Release，快照通过archives目录交接。

v387模型最佳仍为用户报告51.314；52分及MAE平均降低5%尚未实现。公开标签与随机扰动材料明确标为诊断，并记录赛事规则限制，避免被接手者误作模型提分。

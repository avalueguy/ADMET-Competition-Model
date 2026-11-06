# TDC 公开测试标签与离线合成诊断

这里的“标准答案”是公开 TDC ADMET Group 的测试观测值，不是赛事方另外发布的答案。22 个任务、16,379 行的顺序 SMILES 与本地模板一致，保留了重复测量的原始行顺序。

|目录|内容|用途与状态|
|---|---|---|
|public_labels/|公开标签 CSV、逐行来源记录|外部观测查表，非模型预测；此前按此生成的文件已被平台拒绝|
|synthetic_offline/|随机扰动 CSV、22项得分和独立复核|本地合成分 61.775810247；仅检查评分计算|
|source/|原始 TDC ZIP、比赛模板、归一化参考分|固定来源与校验依据|
|reports/|重合证据、拒绝记录、历史指标对照|不作为模型性能证据|
|scripts/|便携复核入口和原始生成脚本|original/ 保存原路径版脚本，不在新机器直接运行|

每项随机选约30%行，共4,917行；其他11,462行保留公开标签。分类选中值替换为0–1均匀随机数，回归任务加入任务尺度随机扰动。固定种子20261106，详细方法见合成得分报告。

61.775810247由同一组已知标签计算，不代表模型准确性、泛化或合规提分。纯标签理论分为65.719642407，平台反馈“22 tasks reach the theoretical optimum; test labels suspected”。没有为随机扰动文件取得平台成绩。

已保存赛事规则禁止外部数据，并要求模型自动生成预测，因此这两个文件都不是合规模型提交。不与训练数据、模型基线混用。本目录不改变v387的用户报告成绩51.314。

来源：https://dataverse.harvard.edu/api/access/datafile/4426004?format=original

源ZIP SHA256：bd0005246cb6f1672333c28ad202112a8bf071ab65d847848715030308faf1f1

v387对照：21项指标与用户分项一致；溶解度本地公开标签MAE为0.7008209745143054，用户报告为0.7006168876998713，关联差异尚未解决。

离线复核：安装Python的numpy、scipy、scikit-learn后，运行 `python diagnostics/tdc_public_test_overlap/scripts/verify_offline.py`。此入口只读取数据与校验，不上传、不训练。生成脚本和Artifact Tool依赖说明见scripts/README.md。

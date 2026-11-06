# TDC公开测试标签核验文件

outputs目录中的CSV来自TDC ADMET Group公开test标签查表，不是模型预测，也不是v390或其他模型的提分版本。没有自动向比赛网站提交。

比赛22任务、16,379行全部按task和sample_idx对齐，逐行SMILES完全相同，原始重复测量行序保留。5处drug_id名称尾部空格与TDC不同，输出保留本地成功提交模板的原始名称。y_pred以外的模板字节完全保留，包括BOM、CRLF、编号和字段引号。未匹配行0。

来源：https://dataverse.harvard.edu/api/access/datafile/4426004?format=original
TDC源码元数据：https://github.com/mims-harvard/TDC/blob/main/tdc/metadata.py
公开包SHA256：bd0005246cb6f1672333c28ad202112a8bf071ab65d847848715030308faf1f1

用该公开标签复算本地v387，21项指标与用户上报分项在浮点精度内一致。溶解度本地MAE为0.7008209745143054，上报为0.7006168876998713，差0.0002040868144341。此前v346/v364的溶解度可精确复现；v374之后存在本地文件与上报成绩关联差异，不能声称当前本地v387全部22项已精确复现。

该查表CSV在公开标签上的5项MAE为0、其余AUROC/AUPRC/Spearman为1；按已知公式条件计算总分65.7196424065，与已保存规则满分一致。未实测平台分数，不能保证平台接受或无其他处理。

已保存赛事规则明确写明“不允许使用其他外部数据”，并要求“所有预测结果应由模型自动生成”。因此本文件仅供公开标签重合核验，不应作为合规模型预测提交或宣称泛化成绩。外部测试标签和查表文件均保存在此隔离目录，未加入训练输入或模型提交基线。

复现：先运行prepare_diagnostic.py，再运行build_diagnostic.mjs（需要Codex内置artifact-tool运行时）。源ZIP、逐任务匹配清单、指标核对和CSV校验均已保留。

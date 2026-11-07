# Transformer 与22项独立专家方案评估

当前官方显示51.240，精确分项均值51.239528409。建议结合两个思路：共享原子/键分子Transformer表示，22项分别建立实验、训练头和适配器、选择模型及融合；HIA维持当前输出。方案已设计，尚未实现新训练器或启动训练。

## 可行性结论

22个独立实验项目可行；22套完全独立的大Transformer不适合本数据规模。hERG457、DILI331、CYP底物约465–468条训练标签，容易过拟合。CYP抑制约8千–9千条，更适合先检验表征学习。独立模型增加任务专用性，但跨任务共享能给小样本提供信息。

现有官方训练数据共53,695条标签行、34,610条不同原始SMILES；train+valid共65,430条标签行、40,229条不同原始SMILES。这里不是规范化去重分子计数，更不是百万级预训练数据。每个验证编码器只能使用其拟合集子集。

既有失败证据：v58做过ChemBERTa微调，未采用序列融合；v238是指纹视图token Transformer，六项未通过开发筛选；v159也试过多任务迁移和梯度冲突处理。因此“换Transformer”“独立任务”“排序损失”本身均不是新的提分证据。原子/键层的编码与拟合集自监督适配，是本方案要检验的新组合假设。

## 数据范围

规则核对状态：LIVE_PUBLIC_API_CHECKED。官网公开接口仍载明“不允许使用其他外部数据”。本新路线采用官方拟合数据学习表示，不新增外部预训练权重。规则没有明确授权外部权重；如以后考虑ChemBERTa/MoLFormer/Uni-Mol等，先取得明确适用解释。历史v361含早期预训练来源的事实不等于新路线已获授权。

## 模型设计

候选主干：6层、192维、6头的原子/键Graph Transformer，加入键类型、图距离、电荷/环特征的注意力偏置。参数预算暂定约2–5百万，待实现实测。编码器在每个外层拟合集独立做掩码原子/键重建与同分子不同视图一致性训练；查询骨架从结构预训练和所有任务标签中一起排除。

小样本任务冻结主干，训练rank8低秩适配器及小头；较大任务比较冻结与解冻末两层。各任务从共享初始化复制模型，随后独立优化，不在一个目标下互相拖动。其他官方测量上下文作为可选独立分支，目标值及掩码严格排除，并与纯结构方案分别报告。

若新表示有增量，再考虑与原标签检索或现有树/核专家融合；这些机制已有历史实验，不将重新组合自动视为创新或收益。3D仅作为第二阶段消融，构象生成与实际溶液构象的不确定性需评估。

## 22项独立实验设计

P1为首批六项；P2在首批证明表示有效后扩展；P3预算较低；KEEP维持当前输出。具体算法是候选假设，不是已训练结果。

|任务|train/valid|优先级|专用模型|目标与限制|
|---|---:|---|---|---|
|ames|4684/1137|P2|原子/键Graph Transformer＋反应性片段残差头|BCE＋成对排序；同一分子等价视图一致性；中等样本，解冻末2层；与SVC专家比较互补性|
|bbb_martins|1421/203|P3|冻结Graph Transformer＋脂溶性/极性描述符小头|BCE＋排序；当前AUROC .929，数学总分空间有限；小适配器|
|bioavailability_ma|448/64|P2|Graph Transformer适配器＋吸收/代谢上下文分支|BCE＋排序；448条训练；上下文分支独立消融；不将HIA/PPBR测量视为必然可用|
|caco2_wang|637/91|P1|Graph Transformer＋理化性质残差头|原尺度L1/中位数回归；可选成对性质差辅助项；637条训练；冻结主干，只训练小适配器和头；成对差路线需单独证明增量|
|clearance_hepatocyte_az|848/122|P1|Graph Transformer适配器＋清除率任务专用头|Gaussian-rank回归＋成对排序；848条训练；同微粒体任务共享只作迁移初始化，独立微调；测量上下文独立消融|
|clearance_microsome_az|770/111|P2|Graph Transformer适配器＋微粒体清除率专用头|Gaussian-rank回归＋成对排序；770条训练；不直接强制与肝细胞标签数值相同|
|cyp2c9_substrate_carbonmangels|467/67|P2|底物家族迁移的Graph Transformer适配器|BCE＋可微AP辅助项的单独消融；仅91个训练正例；不以重采样后更高AP当原分布收益；保留当前强基线|
|cyp2c9_veith|8463/1210|P2|CYP抑制家族Graph Transformer＋任务头|BCE为对照，另测AP辅助损失；8463条训练；可解冻末2层；比较结构Transformer相对当前核专家的增量|
|cyp2d6_substrate_carbonmangels|465/67|P2|底物家族迁移适配器＋条件测量小分支|BCE＋AP辅助项单独消融；465条训练、134正例；小头、双种子；不能照搬大抑制任务全量微调|
|cyp2d6_veith|9191/1313|P1|Graph Transformer＋CYP2D6抑制专用头|BCE对照＋AP辅助目标消融；9191条训练，最适合首批检查Transformer学习能力的任务之一|
|cyp3a4_substrate_carbonmangels|468/67|P2|底物家族Graph Transformer小适配器|BCE＋成对排序；468条训练；必须对比当前恢复分支，不能重用已退步v354组合|
|cyp3a4_veith|8628/1233|P2|抑制家族Graph Transformer＋CYP3A4头|BCE＋AP辅助目标消融；8628条训练；现有AUPRC .899，优先级低于CYP2D6抑制|
|dili|331/48|P3|冻结Graph Transformer＋反应性片段小头|BCE＋排序；331条训练，官方AUROC .935；谨慎分配预算|
|half_life_obach|465/67|P2|Graph Transformer适配器＋半衰期专用双目标头|log1p与Gaussian-rank双目标＋排序；465条训练；当前双目标树已有效，先检查新增表示互补性|
|herg|457/66|P1|Graph Transformer小适配器＋电荷/芳香片段头|BCE＋成对排序；457条训练；新表示先冻结；旧模型已用理化特征，单加描述符不算新机制|
|hia_hou|403/58|KEEP|保留v361完整输出；仅建立独立登记|不启动新训练；官方AUROC已1.0，没有正向分数空间|
|ld50_zhu|3970/1937|P2|Graph Transformer＋毒性专用回归头|原尺度L1为主；稳健尾部处理单独消融；3970条训练；先核对单位和标签变换，不能仅追求训练集RMSE|
|lipophilicity_astrazeneca|2940/420|P1|Graph Transformer＋残差MAE头|原尺度L1；以现有物理/检索专家为对照；2940条训练；验证表示互补，校准仅用拟合/内层预测|
|pgp_broccatelli|851/122|P3|冻结Graph Transformer＋P-gp小头|BCE＋排序；851条训练，官方AUROC .950；极限总分收益仅约.058|
|ppbr_az|1952/279|P2|Graph Transformer＋有界中位数回归头|原尺度L1；[0,100]输出或固定边界裁剪；1952条训练；直接优化PPBR误差，不能以变换尺度的损失替代最终MAE|
|solubility_aqsoldb|5045/2940|P1|Graph Transformer＋溶解度MAE头|原尺度L1；重复测量不确定性作为可选消融；5045条训练；不能把结构增强当作新增独立标签；标签检索已试过，必须检验新增结构表示|
|vdss_lombardo|791/113|P2|Graph Transformer适配器＋VDss排序头|Gaussian-rank回归＋成对排序；791条训练；理化性质/PPBR关系作为辅助，不强加未经证实的确定方程|

## 实验和项目组织

采用同一代码仓库下22个任务实验目录，每项独立配置、基线哈希、开发预测、选型记录、检查点、最终预测与报告；共享数据加载、骨架划分、编码器和CSV组装代码。最终按task/sample_idx映射组装模板，只写y_pred。无需复制22份代码，也无需创建22个Codex聊天。

首批六项：水溶解度、脂溶性、Caco-2、CYP2D6抑制、肝细胞清除率、hERG。覆盖MAE、AUPRC、Spearman和AUROC，也覆盖数据较充足与小样本场景。

分阶段消融：
1. A: current v361 output plus reproducible current structure control; preserve upstream OOF limitations
2. B: graph Transformer from scratch, target head only
3. C: fit-only self-supervised graph Transformer plus target adapter; compare to B
4. D: C plus target-family supervised initialization; compare to C, global query groups excluded
5. E: strongest frozen developmental C/D plus optional target-excluded context, separate hypothesis
6. F: fixed coarse blend of expert with v361, include zero weight; freeze before confirmation

使用开发fold0/1做有界选择，冻结后检查fold2及官方valid，保留原严格验收规则；实验候选可按既有授权单独标明，但不能改写严格拒绝记录。所有公开标签长期复用，新骨架划分与bootstrap不是全新独立样本。旧完整集成OOF缺失的任务需保留“代理对照”标注，不能借新方案抹去这一限制。

每个任务保留基线和有限个假设，以骨架验证及增量为停止条件；“一直训练到最好”不存在可验证终点，反复观察同一验证集会选中噪声。各项独立训练可以独立选择，但不能承诺隐藏测试每项均不退步。

## 计分情景与大幅提升门槛

下表是把假设原始指标代入已由成绩单复核的计分关系所得的条件算术，**不是模型成绩预测、置信区间或成功概率**。AUROC/AUPRC增加为绝对值，MAE下降为相对比例，Spearman增加为绝对值；达到1的任务封顶。每一行假设相应组内全部任务都达到改进且无回退。

|情景|5项MAE下降|未满分类指标增加|4项Spearman增加|条件总分|
|---|---:|---:|---:|---:|
|未发现可用新信号，保留v361|0.0%|0.000|0.000|51.240|
|小幅联合改善情景|3.0%|0.005|0.020|51.550|
|冲52的联合改善情景|8.0%|0.015|0.050|52.096|
|进取联合改善情景|12.0%|0.025|0.080|52.589|
|53分以上需要的强改善情景|20.0%|0.030|0.100|53.255|
|54分以上需要的极强改善情景|25.0%|0.050|0.150|54.057|

未发现有效新信号时维持51.240是可能结果；未经有效选型的新提交也可能低于基线。52是应首先检验的里程碑，53是需要更强证据的进取目标，54及以上不能作为现阶段可信预期。

只提升5个MAE任务、其余17项固定时：
- 达到52：五项MAE均需下降约12.59%。
- 达到53：五项MAE均需下降约26.99%。
- 达到54：五项MAE均需下降约39.41%。

hERG、BBB、DILI和P-gp即使全部达到AUROC=1，总分最多也只增加约0.387。低平台分不等于大幅可提升空间；HIA已1.0，无法贡献新增分数。五项MAE应得到重点预算。

换算关系：MAE分项=100*reference/(raw+reference)；分类分项=100*raw/(raw+reference)；Spearman分项=100*(raw+1)/(raw+reference+2)；总分为22项均值。当前22行复核误差不超过0.00000051。该关系来自已观测成绩的经验复核，不冒称已获取官方完整计分文档。

## 资源与交付边界

本次没有检查服务器当前GPU状态，不能给可靠工期。先对代表任务测实测吞吐和显存，再排预算。21项可训练任务×4个开发分区×2种子=168次专家拟合/假设，尚不含编码器预训练、消融、对照和最终重训；因此不一次盲跑22套大模型。

详细数值、样本量、任务设计和输入哈希见 [JSON](transformer22_plan.json)，最新规则核对见 [记录](transformer22_rule_check.json)。本轮交付方案及算术分析，没有启动新训练、生成新提交或创建新聊天。

# 新服务器优化实验结果

完成时间（香港）：2026-11-05T15:10:48.275679+08:00。

官方已知最高仍为 v364，51.269。所有新训练都在用户指定的端口25870服务器 RTX3090 上完成。

|阶段|任务|结果|
|---|---|---|
|v368|ames, clearance_microsome_az, half_life_obach, vdss_lombardo|无符合条件的新提交|
|v369|bioavailability_ma, cyp2c9_substrate_carbonmangels, cyp2d6_substrate_carbonmangels, cyp3a4_substrate_carbonmangels|无符合条件的新提交|
|v370|caco2_wang, ld50_zhu, lipophilicity_astrazeneca, ppbr_az, solubility_aqsoldb|无符合条件的新提交|

本轮三组实验均未找到符合条件的新候选，没有生成新CSV。可继续使用此前官方最好文件：

`admet_submissions/v364_graph_transformer_experimental/我从山北来带着兰花草-小分子药物预测技术创新赛-20261104.csv`

SHA256：`777eca282b59101463fe02f71802c24502a87ba7352482b24a64943ef81ede79`。

v368：Graph Transformer从零、自监督和任务家族迁移。AMES和半衰期开发改善，但valid退步；微粒体清除率和VDss开发未通过。
v369：生物利用度和CYP底物小任务适配器。生物利用度开发改善，但fold2和valid退步；三项CYP底物开发未通过。
v370：Graph Transformer增加4096维count指纹分支，比较AdamW和rho=.05的SAM，覆盖五项MAE任务。详细四分区结果见对应报告。

不能把新增架构和更多组合视为必然提分。失败时保留原预测，未根据确认集反复换配方。新测试使用了长期复用的公开验证标签，历史控制预测部分是代理OOF。

下一步优先核查训练与隐藏测试分布差异、测量单位及重复标签质量，建立更可信的分组对照；随后针对可解释误差训练有限任务专家。缺少新增信号时，继续堆叠同类Transformer不支持大幅提分预期。

远端每个模型检查点均保留；本地保存输入、源码、预测、哈希清单及独立重放证据。新服务器没有前一服务器的旧训练目录；旧v364/v366最终权重已在此前GitHub交接包保留。密码不在文件中。

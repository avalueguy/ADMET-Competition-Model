# v387 结果：完整 v385 基线

官方已知 v385 为51.308，新实验分数待用户提交。所有拟合在指定25870远端。

|任务|配置|权重|fold0 MAE降幅|fold1|fold2|valid|结论|
|---|---|---:|---:|---:|---:|---:|---|
|Caco2|gaussian_full_C30_r2|0.1|+0.094%|+0.389%|-0.466%|+0.089%|确认未通过|
|LD50|gaussian_full_C3_r0|0.1|-0.091%|-0.189%|未运行|未运行|开发未通过|
|脂溶性|gaussian_full_C3_r2|0.1|+0.411%|+0.410%|+0.205%|+0.843%|实验采用|
|PPBR|gaussian_full_C3_r8|0.1|+0.702%|-0.033%|未运行|未运行|开发未通过|
|溶解度|gaussian_full_C3_r0|0.1|+0.184%|-0.443%|未运行|未运行|开发未通过|
|肝细胞清除率|gaussian_full_C3_r0|0.5|+4.187%|+6.369%|+3.265%|+7.486%|确认未通过|
|肝微粒体清除率|gaussian_full_C3_r0|0.1|+1.892%|+1.696%|未运行|未运行|开发未通过|
|半衰期|gaussian_full_C3_r0|0.25|+15.224%|+12.329%|+3.497%|-9.547%|确认未通过|
|VDss|gaussian_full_C3_r0|0.1|+3.973%|+1.350%|未运行|未运行|开发未通过|

未采用原因：
- Caco2：fold2:MAE不改善
- LD50：fold0:MAE不改善；fold1:MAE不改善；开发门槛未过，未运行确认
- PPBR：fold1:MAE不改善；开发门槛未过，未运行确认
- 溶解度：fold1:MAE不改善；开发门槛未过，未运行确认
- 肝细胞清除率：fold2:Spearman不改善
- 肝微粒体清除率：fold1:Spearman不改善；开发门槛未过，未运行确认
- 半衰期：valid:MAE不改善；fold2:Spearman不改善
- VDss：fold0:Spearman不改善；fold1:Spearman不改善；开发门槛未过，未运行确认

324次开发求解全部完成，再固定唯一候选作确认；没有使用确认结果改选候选。独立重放、查询标签扰动、输入/源码/模型哈希通过。

9项平均相对MAE降幅：{'fold0': 0.0004567329018143947, 'fold1': 0.000455353776391914, 'fold2': 0.00022778403254324512, 'valid': 0.0009368330454599932}。5项直接MAE计分平均：{'fold0': 0.0008221192232659104, 'fold1': 0.0008196367975054451, 'fold2': 0.0004100112585778412, 'valid': 0.0016862994818279877}。拒绝项计零。

9项平均5%目标已达到：False。

实验提交：`admet_submissions/v387_v385_correlated_metric_experimental/我从山北来带着兰花草-小分子药物预测技术创新赛-20261106.csv`。新官方成绩未知。

Repeatedly used public partitions, historical controls partly proxy. Four ranking MAEs use calibrated references. Positive direction is not proof of hidden-test improvement.

本轮独立比较完整v385，不与其他未评分候选叠加。

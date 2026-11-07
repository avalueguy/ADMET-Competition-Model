# v386 结果：完整 v385 基线

官方已知 v385 为51.308，新实验分数待用户提交。所有拟合在指定25870远端。

|任务|配置|权重|fold0 MAE降幅|fold1|fold2|valid|结论|
|---|---|---:|---:|---:|---:|---:|---|
|Caco2|sqrt_median_mono02|0.5|+1.697%|+1.597%|+0.979%|-0.433%|确认未通过|
|LD50|tanimoto_median_mono02|0.1|+0.022%|+0.096%|+0.004%|-0.347%|确认未通过|
|脂溶性|sqrt_gaussian_rank_mono02|0.1|+0.204%|+0.317%|+0.069%|+0.443%|实验采用|
|PPBR|sqrt_gaussian_rank_mono02|0.1|+0.472%|-0.074%|未运行|未运行|开发未通过|
|溶解度|sqrt_median_mono02|0.1|+0.125%|+0.083%|-0.742%|+0.564%|确认未通过|
|肝细胞清除率|sqrt_gaussian_rank_mono10|0.75|+11.216%|+10.518%|+4.783%|+2.974%|确认未通过|
|肝微粒体清除率|sqrt_median_raw|0.1|+1.537%|+1.343%|未运行|未运行|开发未通过|
|半衰期|sqrt_gaussian_rank_mono02|0.75|+44.190%|+35.075%|+4.275%|-32.387%|确认未通过|
|VDss|sqrt_median_raw|0.1|+3.789%|+0.174%|未运行|未运行|开发未通过|

未采用原因：
- Caco2：valid:MAE不改善
- LD50：valid:MAE不改善
- PPBR：fold1:MAE不改善；开发门槛未过，未运行确认
- 溶解度：fold2:MAE不改善
- 肝细胞清除率：fold2:Spearman不改善；valid:Spearman不改善
- 肝微粒体清除率：fold1:Spearman不改善；开发门槛未过，未运行确认
- 半衰期：valid:MAE不改善；fold2:Spearman不改善
- VDss：fold0:Spearman不改善；fold1:Spearman不改善；开发门槛未过，未运行确认

216次开发基础SVR、108次开发校准LP全部完成，再固定唯一候选作确认；没有使用确认结果改选候选。独立重放、查询标签扰动、输入/源码/模型哈希通过。

9项平均相对MAE降幅：{'fold0': 0.00022680879549530627, 'fold1': 0.0003524284097991925, 'fold2': 7.690551231735848e-05, 'valid': 0.0004926840760802742}。5项直接MAE计分平均：{'fold0': 0.0004082558318915513, 'fold1': 0.0006343711376385465, 'fold2': 0.00013842992217124526, 'valid': 0.0008868313369444936}。拒绝项计零。

9项平均5%目标已达到：False。

实验提交：`admet_submissions/v386_v385_oof_monotone_experimental/我从山北来带着兰花草-小分子药物预测技术创新赛-20261106.csv`。新官方成绩未知。

Repeatedly used public partitions, historical controls partly proxy. Four ranking MAEs use calibrated references. Positive direction is not proof of hidden-test improvement.

本轮独立比较完整v385，不与其他未评分候选叠加。

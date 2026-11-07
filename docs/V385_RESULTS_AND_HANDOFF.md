# v385 结果：完整 v382 基线

官方已知 v382 为51.299，新实验分数待用户提交。所有拟合在指定25870远端。

|任务|配置|权重|fold0 MAE降幅|fold1|fold2|valid|结论|
|---|---|---:|---:|---:|---:|---:|---|
|Caco2|sqrt_native_C3|0.25|+0.124%|+0.293%|+0.275%|-0.260%|确认未通过|
|LD50|sqrt_gaussian_rank_C30|0.1|-0.089%|-0.144%|未运行|未运行|开发未通过|
|脂溶性|sqrt_gaussian_rank_C30|0.1|+0.140%|+0.275%|-0.077%|+0.418%|确认未通过|
|PPBR|sqrt_gaussian_rank_C30|0.1|+0.719%|+0.114%|+0.107%|+0.143%|实验采用|
|溶解度|sqrt_native_C3|0.1|+0.140%|-0.008%|未运行|未运行|开发未通过|
|肝细胞清除率|sqrt_gaussian_rank_C30|0.75|+11.401%|+10.500%|+4.836%|+3.119%|确认未通过|
|肝微粒体清除率|sqrt_gaussian_rank_C3|0.25|+3.661%|+3.470%|+6.985%|+6.790%|确认未通过|
|半衰期|tanimoto_gaussian_rank_C30|0.75|+44.564%|+35.019%|+3.936%|-32.217%|确认未通过|
|VDss|tanimoto_gaussian_rank_C30|0.5|+19.070%|+7.870%|+29.030%|+19.735%|确认未通过|

296次开发求解全部完成，再固定唯一候选作确认；没有使用确认结果改选候选。独立重放、查询标签扰动、输入/源码/模型哈希通过。

9项平均相对MAE降幅：{'fold0': 0.0007991373793543566, 'fold1': 0.00012617061160694065, 'fold2': 0.00011882694540894186, 'valid': 0.00015842131614626274}。5项直接MAE计分平均：{'fold0': 0.001438447282837842, 'fold1': 0.00022710710089249314, 'fold2': 0.00021388850173609535, 'valid': 0.0002851583690632729}。拒绝项计零。

9项平均5%目标已达到：False。

实验提交：`admet_submissions/v385_v382_warped_median_experimental/我从山北来带着兰花草-小分子药物预测技术创新赛-20261106.csv`。新官方成绩未知。

Repeatedly used public partitions, historical controls partly proxy. Four ranking MAEs use calibrated references. Positive direction is not proof of hidden-test improvement.

v384没有合格更新，v385仍直接比较完整v382。

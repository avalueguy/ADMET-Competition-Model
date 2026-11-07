# v384 结果：完整 v382 基线

官方已知 v382 为51.299，新实验分数待用户提交。所有拟合在指定25870远端。

|任务|配置|权重|fold0 MAE降幅|fold1|fold2|valid|结论|
|---|---|---:|---:|---:|---:|---:|---|
|Caco2|laplace_C3_r8|0.1|-0.535%|-0.148%|未运行|未运行|开发未通过|
|LD50|laplace_C30_r0|0.1|-0.146%|-0.161%|未运行|未运行|开发未通过|
|脂溶性|laplace_C30_r8|0.1|-1.099%|-0.243%|未运行|未运行|开发未通过|
|PPBR|laplace_C3_r5|0.1|-1.159%|-1.069%|未运行|未运行|开发未通过|
|溶解度|laplace_C3_r2|0.1|+0.215%|-0.743%|未运行|未运行|开发未通过|
|肝细胞清除率|laplace_C3_r0|0.25|+1.909%|+4.194%|+2.816%|+5.852%|确认未通过|
|肝微粒体清除率|laplace_C3_r0|0.1|+1.584%|+1.930%|未运行|未运行|开发未通过|
|半衰期|laplace_C3_r0|0.1|+6.063%|+4.838%|未运行|未运行|开发未通过|
|VDss|laplace_C3_r0|0.1|+4.038%|+1.221%|未运行|未运行|开发未通过|

324次开发求解全部完成，再固定唯一候选作确认；没有使用确认结果改选候选。独立重放、查询标签扰动、输入/源码/模型哈希通过。

9项平均相对MAE降幅：{'fold0': 0.0, 'fold1': 0.0, 'fold2': 0.0, 'valid': 0.0}。5项直接MAE计分平均：{'fold0': 0.0, 'fold1': 0.0, 'fold2': 0.0, 'valid': 0.0}。拒绝项计零。

9项平均5%目标已达到：False。

实验提交：`None`。新官方成绩未知。

Repeatedly used public partitions, historical controls partly proxy. Four ranking MAEs use calibrated references. Positive direction is not proof of hidden-test improvement.

v383为独立的旧脂溶性恢复文件，不混入本轮主实验。

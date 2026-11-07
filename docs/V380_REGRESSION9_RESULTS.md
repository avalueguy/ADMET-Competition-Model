# v380 九项回归专家结果

官方基线：用户显示51.271；关联v374本地文件。全部拟合仅指定25870远端CUDA服务器完成。

|任务|开发最优|权重|fold0 MAE降幅|fold1 MAE降幅|fold2 MAE降幅|valid MAE降幅|结果|
|---|---|---:|---:|---:|---:|---:|---|
|caco2_wang|sqrt_polynomial_C3_native|0.25|+0.124%|+0.293%|+0.275%|-0.260%|确认失败|
|ld50_zhu|sqrt_polynomial_C3_native|0.1|-0.163%|-0.014%|未运行|未运行|开发失败|
|lipophilicity_astrazeneca|sqrt_polynomial_C30_native|0.1|+0.184%|+0.202%|+0.303%|+0.635%|实验候选|
|ppbr_az|sqrt_polynomial_C30_native|0.1|-0.601%|-0.805%|未运行|未运行|开发失败|
|solubility_aqsoldb|sqrt_polynomial_C3_native|0.1|+0.140%|-0.008%|未运行|未运行|开发失败|
|clearance_hepatocyte_az|sqrt_polynomial_C30_weighted_log|0.5|+6.314%|+8.912%|+5.774%|+2.857%|确认失败|
|clearance_microsome_az|sqrt_polynomial_C3_weighted_log|0.25|+4.319%|+3.779%|+6.175%|+6.809%|确认失败|
|half_life_obach|sqrt_polynomial_C30_weighted_log|0.25|+14.939%|+12.200%|+3.141%|-8.265%|确认失败|
|vdss_lombardo|hellinger_gaussian_C3_weighted_log|0.1|+4.490%|+1.566%|+6.600%|+4.241%|确认失败|

采用全部任务后的9项平均相对MAE降幅：{'fold0': 0.0002042872524576704, 'fold1': 0.00022486534984988957, 'fold2': 0.0003364801906713555, 'valid': 0.0007050658044118001}。5项官方MAE平均：{'fold0': 0.00036771705442380676, 'fold1': 0.00040475762972980124, 'fold2': 0.0006056643432084399, 'valid': 0.0012691184479412404}。
拒绝任务计入零改善，没有只对获胜任务求平均。四个分区均达到9项平均5%：False。
四项排序任务的MAE为拟合标签分位数校准参考；该误差基线不是旧模型原物理单位预测。排序提升另行检查。
本轮实验文件：`admet_submissions/v380_regression9_experimental/我从山北来带着兰花草-小分子药物预测技术创新赛-20261105.csv`，官方成绩未知。
置信区间、单任务原MAE和排序增益见JSON。原严格拒绝保留，不能把实验方向一致误当稳定官方收益。

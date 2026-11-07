# v378 九项回归专家结果

官方基线：用户显示51.271；关联v374本地文件。全部拟合仅指定25870远端CUDA服务器完成。

|任务|开发最优|权重|fold0 MAE降幅|fold1 MAE降幅|fold2 MAE降幅|valid MAE降幅|结果|
|---|---|---:|---:|---:|---:|---:|---|
|caco2_wang|feature_transformer_n800|0.1|+0.436%|-0.256%|未运行|未运行|开发失败|
|ld50_zhu|retrieval_transformer_n2400|0.1|-0.148%|-0.128%|未运行|未运行|开发失败|
|lipophilicity_astrazeneca|feature_transformer_n800|0.1|-0.459%|-0.180%|未运行|未运行|开发失败|
|ppbr_az|retrieval_transformer_n800|0.1|+0.407%|-0.265%|未运行|未运行|开发失败|
|solubility_aqsoldb|retrieval_transformer_n800|0.1|+0.256%|-0.131%|未运行|未运行|开发失败|
|clearance_hepatocyte_az|retrieval_transformer_n4800|0.25|+4.559%|+3.764%|+3.453%|+4.346%|确认失败|
|clearance_microsome_az|feature_transformer_n4800|0.1|+1.916%|+1.633%|+3.367%|+2.876%|确认失败|
|half_life_obach|feature_transformer_n800|0.1|+5.576%|+4.659%|未运行|未运行|开发失败|
|vdss_lombardo|feature_transformer_n800|0.1|+4.727%|+1.183%|未运行|未运行|开发失败|

采用全部任务后的9项平均相对MAE降幅：{'fold0': 0.0, 'fold1': 0.0, 'fold2': 0.0, 'valid': 0.0}。5项官方MAE平均：{'fold0': 0.0, 'fold1': 0.0, 'fold2': 0.0, 'valid': 0.0}。
拒绝任务计入零改善，没有只对获胜任务求平均。四个分区均达到9项平均5%：False。
四项排序任务的MAE为拟合标签分位数校准参考；该误差基线不是旧模型原物理单位预测。排序提升另行检查。
本轮没有合格更新，没有生成新CSV。
置信区间、单任务原MAE和排序增益见JSON。原严格拒绝保留，不能把实验方向一致误当稳定官方收益。

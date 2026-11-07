# v101 官方分任务成绩单（用户提供）

总分：50.932（2026-10-26）

## 优先优化任务

| 任务 | 指标 | 分任务得分 | 原始指标 | 参考值 |
| --- | --- | ---: | ---: | ---: |
| cyp3a4_substrate_carbonmangels | AUROC | 48.897017 | 0.693942 | 0.725249 |
| caco2_wang | MAE | 49.287576 | 0.291847 | 0.283647 |
| herg | AUROC | 49.651295 | 0.851399 | 0.863358 |
| solubility_aqsoldb | MAE | 49.831244 | 0.743445 | 0.738443 |
| ames | AUROC | 49.980787 | 0.871275 | 0.871945 |
| bbb_martins | AUROC | 50.02653 | 0.928764 | 0.927779 |

## 已有优势任务

| 任务 | 指标 | 分任务得分 | 原始指标 | 参考值 |
| --- | --- | ---: | ---: | ---: |
| cyp2c9_substrate_carbonmangels | AUPRC | 56.936749 | 0.586209 | 0.44337 |
| bioavailability_ma | AUROC | 52.391603 | 0.794812 | 0.722248 |
| lipophilicity_astrazeneca | MAE | 52.375585 | 0.429997 | 0.472895 |
| ld50_zhu | MAE | 52.05411 | 0.555475 | 0.603071 |
| ppbr_az | MAE | 52.047135 | 6.753957 | 7.330617 |
| half_life_obach | Spearman | 51.844389 | 0.605632 | 0.49139 |

策略：优先训练 cyp3a4_substrate、caco2_wang、herg、solubility_aqsoldb；对 bioavailability_ma、cyp2c9_substrate_carbonmangels、ld50_zhu、lipophilicity_astrazeneca、ppbr_az 使用冻结 v101 预测，不做无必要扰动。

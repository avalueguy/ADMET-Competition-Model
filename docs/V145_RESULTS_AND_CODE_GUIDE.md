# v145 最高分版本、代码与后续优化成果

整理日期：2026-10-31。本说明依据本地已核验记录，不访问比赛平台，不将实验编号当作可提交版本。

## 1. 当前能确认的成果

- 官方最佳：v145，显示总分 **51.166**；22 项分数均值 51.1661275；尚未达到 52。
- 官方记录时间：2026-10-27 17:22。成绩单与交付记录按分项匹配；平台未公开文件哈希，不能证明平台上传字节哈希。
- CSV：`admet_submissions/v145_feedback_repair/我从山北来带着兰花草-小分子药物预测技术创新赛-20261027.csv`。
- SHA256：`3ffd68785b3fc321c8e81a32b23a09cf2493ff26240555a3472a315c572706e7`。
- 16,379 条预测、22 个任务；本地交付记录验证 BOM/CRLF、Drug_ID 原字符串及模板非预测字节。
- 本次 GitHub 上传不包含该 CSV 或权重；它是代码与验证证据交接，不能只克隆源码就宣称已完整复现。

## 2. v145 是怎样构成的

v130 官方显示 51.146。v144 更新五项后显示 51.099，主要损失来自 CYP3A4 底物和肝细胞清除率。v145 保留 v144 的 AMES、CYP2C9 抑制、CYP2D6 抑制输出，其余 19 项恢复/沿用 v130。它按整任务选择既有输出，没有逐样本手调、重训练或重新扫描融合权重。

相对 v130，AMES 变化 1,457 条、CYP2C9 抑制 2,419 条、CYP2D6 抑制 2,626 条，总计 6,502 条。更完整的来源哈希见 [交付记录](v145_local_delivery.json)。

## 3. 22 项官方成绩

| 任务 | 指标 | 平台分项 | 原始指标 |
| --- | --- | ---: | ---: |
| ames | AUROC | 50.141941 | 0.876909671 |
| bbb_martins | AUROC | 50.032316 | 0.928979049 |
| bioavailability_ma | AUROC | 52.391603 | 0.794812105 |
| caco2_wang | MAE | 50.689546 | 0.275929921 |
| clearance_hepatocyte_az | Spearman | 50.211228 | 0.565208997 |
| clearance_microsome_az | Spearman | 50.495262 | 0.675684544 |
| cyp2c9_substrate_carbonmangels | AUPRC | 56.936749 | 0.586208554 |
| cyp2c9_veith | AUPRC | 51.364323 | 0.834940858 |
| cyp2d6_substrate_carbonmangels | AUPRC | 50.358383 | 0.758324470 |
| cyp2d6_veith | AUPRC | 50.846574 | 0.743944717 |
| cyp3a4_substrate_carbonmangels | AUROC | 50.706702 | 0.746044304 |
| cyp3a4_veith | AUPRC | 50.422543 | 0.899191534 |
| dili | AUROC | 50.715743 | 0.935000000 |
| half_life_obach | Spearman | 52.539161 | 0.650969136 |
| herg | AUROC | 49.832263 | 0.857584683 |
| hia_hou | AUROC | 50.143410 | 1.000000000 |
| ld50_zhu | MAE | 52.054110 | 0.555475369 |
| lipophilicity_astrazeneca | MAE | 52.375585 | 0.429997070 |
| pgp_broccatelli | AUROC | 50.320749 | 0.950479872 |
| ppbr_az | MAE | 52.047135 | 6.753956500 |
| solubility_aqsoldb | MAE | 50.236728 | 0.731483532 |
| vdss_lombardo | Spearman | 50.792751 | 0.714513743 |

分项平台分数不是原始 AUROC/MAE；不能直接按原始指标差值推算总分。MAE 越低越好，其他列出的原始指标越高越好。

## 4. 代码入口与作用

| 路径 | 作用 |
| --- | --- |
| [tools/v145_feedback_repair.py](../tools/v145_feedback_repair.py) | 校验 v130/v144 输出，按三个任务路由并按模板字节构建 v145；属于历史构建脚本，依赖旧 CSV、work/v145_export 等文件，不应直接运行覆盖当前状态。 |
| [tools/v146_current_baseline.py](../tools/v146_current_baseline.py) | 锁定 v145 CSV SHA，重建各原划分控制预测。依赖 v131/v135/v139 及历史 evidence、模型缓存；上游历史 OOF 仍有近似。 |
| [tools/v148_experiment_runner.py](../tools/v148_experiment_runner.py) | 固定开发选择、唯一配置/权重冻结、确认与骨架 bootstrap；没有通过就不产出最终候选。 |
| [tools/v131_current_controls.py](../tools/v131_current_controls.py) | 承载 v130 的任务级历史基线；导入的早期源码为依赖链。 |
| [tools/v135_broad_boosting.py](../tools/v135_broad_boosting.py)、[v139_classification_kernels.py](../tools/v139_classification_kernels.py) | v144 三项保留输出的专家机制。 |
| [tools/v127_low_classification.py](../tools/v127_low_classification.py) | v145 中 hERG/CYP3A4 底物原组件与固定原种子。 |
| [tools/v294_context_uncertainty_audit.py](../tools/v294_context_uncertainty_audit.py) | 仅拟合集的其他测量重复值可行性审查；不是提分验证。 |
| [tools/v295_v145_context_integration.py](../tools/v295_v145_context_integration.py) | 对固定 v145 组件的允许上下文做 16 点确定性经验积分；保留原树、种子、身份与标签，尚未核验最终结果。 |
| [tools/repack_upload_template.py](../tools/repack_upload_template.py)、[verify_submission.py](../tools/verify_submission.py) | 模板字节处理与提交格式核验。 |

源码保留历史字节，部分路径固定；不要为方便而改签名相关源码再声称缓存仍一致。本次可检查源码与证据，但完整重新推理还需要未上传的输入、检查点、缓存、精确环境与 release 依赖。

## 5. 后续实验实际结论

下面来自各次终态 JSON。无合格 CSV 的实验并非新可提交版本；正常结束也不等于模型有效。

| 证据 | 记录状态 | 新 CSV |
| --- | --- | --- |
| [v169_final_status](v169_final_status.json) | `COMPLETE_NO_NEW_SUBMISSION` | 无 |
| [v182_final_status](v182_final_status.json) | `COMPLETE_NO_NEW_SUBMISSION` | 无 |
| [v190_final_status](v190_final_status.json) | `COMPLETE_NO_CONFIRMED_NEW_MODEL` | 无 |
| [v194_final_status](v194_final_status.json) | `COMPLETE_NO_CONFIRMED_NEW_MODEL` | 无 |
| [v198_final_status](v198_final_status.json) | `FAILED_OPTIMIZER_NONCONVERGENCE` | 无 |
| [v201_final_status](v201_final_status.json) | `COMPLETE_NO_CONFIRMED_NEW_MODEL` | 无 |
| [v205_final_status](v205_final_status.json) | `COMPLETE_NO_CONFIRMED_NEW_MODEL` | 无 |
| [v208_final_status](v208_final_status.json) | `COMPLETE_NO_CONFIRMED_NEW_MODEL` | 无 |
| [v211_final_status](v211_final_status.json) | `COMPLETE_NO_CONFIRMED_NEW_MODEL` | 无 |
| [v217_final_status](v217_final_status.json) | `COMPLETE_NO_CONFIRMED_NEW_MODEL` | 无 |
| [v222_final_status](v222_final_status.json) | `COMPLETE_NO_CONFIRMED_NEW_MODEL` | 无 |
| [v225_final_status](v225_final_status.json) | `COMPLETE_NO_CONFIRMED_NEW_MODEL` | 无 |
| [v229_final_status](v229_final_status.json) | `COMPLETE_NO_CONFIRMED_NEW_MODEL` | 无 |
| [v234_final_status](v234_final_status.json) | `COMPLETE_NO_CONFIRMED_NEW_MODEL` | 无 |
| [v239_final_status](v239_final_status.json) | `COMPLETE_NO_CONFIRMED_NEW_MODEL` | 无 |
| [v245_final_status](v245_final_status.json) | `COMPLETE_NO_CONFIRMED_NEW_MODEL` | 无 |
| [v251_final_status](v251_final_status.json) | `COMPLETE_NO_CONFIRMED_NEW_MODEL` | 无 |
| [v257_final_status](v257_final_status.json) | `COMPLETE_NO_CONFIRMED_NEW_MODEL` | 无 |
| [v265_final_status](v265_final_status.json) | `COMPLETE_NO_CONFIRMED_NEW_MODEL` | 无 |
| [v272_final_status](v272_final_status.json) | `COMPLETE_NO_CONFIRMED_NEW_MODEL` | 无 |
| [v279_final_status](v279_final_status.json) | `COMPLETE_NO_CONFIRMED_NEW_MODEL` | 无 |
| [v285_final_status](v285_final_status.json) | `COMPLETE_NO_CSV_IMPLEMENTATION_DEFECT_FOUND` | 无 |
| [v290_final_status](v290_final_status.json) | `COMPLETE_NO_CONFIRMED_NEW_MODEL` | 无 |

关键例子：v271/v272 的肝细胞清除率四处点增益为正，但 fold2/valid 两份骨架 bootstrap 下界均跨零，按原规则拒绝；v284/v285 发现上下文列替换实现缺陷，缺陷结果不能代替原方案结论；v289/v290 修正实现后完成四开发分区、八种子模型，仍无通过任务。其归档 128 成员及依赖本地核验 PASS，且没有新 CSV。

v294 只显示固定模型对重复上下文取值存在敏感性，不读取目标标签或外层查询成绩，不能作为泛化提升证据。v295 对这种输入不确定性进行积分，当前仅交付源码；不宣称完成、提分或新 CSV 已产生。

## 6. 继续优化时保持的验收规则

1. 在原 fold0/1 上选择最差折增益最大的唯一配置与融合权重（0.2 或 0.45），在打开 fold2/valid 前冻结。
2. 开发最小增益必须超过原阈值：MAE 0.012、Spearman 0.008、AUROC/AUPRC 0.003。
3. 四处点增益全部大于零，四处平均增益超过上述实际阈值；fold2 和 valid 两份按骨架成组的 500 次配对 bootstrap 90% 区间下界都必须大于零。
4. 单骨架组确认不可验收；确认失败无回退配置，不以换种子或扫描权重反复试验同一拒绝路线。
5. 通过后还需独立重载/特征重建、融合复算、行序、原 Drug_ID、全部非预测字节、模型与归档 SHA 核验，未改任务保留实际 v145 CSV。

公开标签已反复使用，上游 OOF 不完全等价于完整嵌套重训。上述检查降低风险，不能保证隐藏测试更高；官方分数只以用户后续返回的真实成绩单更新。

## 7. 交付与范围

此次向现有 GitHub 仓库上传源码、分数和验证材料；不改变比赛提交策略。新合格 CSV 仅交用户手动提交；健康训练无实时播报。SSH 密码、令牌、运行凭据不写入或上传。模型权重、数据、特征、CSV 和 release 大归档保留本地/原授权远程环境。

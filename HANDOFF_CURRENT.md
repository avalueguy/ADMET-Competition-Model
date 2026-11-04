# 当前模型交接：v364 最好成绩、v366 实验候选

时间：2026-11-04，Asia/Hong_Kong。本文覆盖截至 v366 的进展，供下一位继续训练；旧交接日志保留为历史证据。

## 当前结论

- 官方最好：v364，用户完整成绩单显示 51.269，22 项均值 51.26918777272727。平台没有提供上传文件哈希，版本关联依据连续反馈和分项吻合。
- 最新候选：v366，已完成训练、重载、独立算术、模板、逐行分子顺序和归档核验；官方分数未知，等待手动提交反馈。
- 目标 52 尚未达到。没有证据支持换 Transformer 后能直接飞跃几分。
- v366 只改 LD50 1,478 行；其余 21 项预测字符串及全部非预测字节与 v364 相同。
- 所有新训练均在指定远端 RTX 3090 完成。目前该轮训练已结束，无待完成训练任务。

## 两个应保留的 CSV

恢复包解开后：

|用途|路径|SHA256|
|---|---|---|
|当前官方最好，回退基线|`admet_submissions/v364_graph_transformer_experimental/我从山北来带着兰花草-小分子药物预测技术创新赛-20261104.csv`|`777eca282b59101463fe02f71802c24502a87ba7352482b24a64943ef81ede79`|
|最新实验候选|`admet_submissions/v366_graph_transformer_experimental/我从山北来带着兰花草-小分子药物预测技术创新赛-20261104.csv`|`45982f798555d88f2c934d187c709091db02841e257c1f2450b470b9a02c04fc`|

用户已授权稳定性尚未完全确认的实验 CSV，严格拒绝记录仍必须保留。不得把实验授权理解为任意选型或隐藏测试保证。

## 从 GitHub 恢复

下载 `admet-v366-source.zip`、所有 `admet-v366-runtime.tar.gz.partNNN`、`handoff-manifest-v366.json`、`restore_v366_handoff.py`，置于同一文件夹。源码 ZIP 保留项目根目录相对路径；分卷是单个 gzip tar 的二进制切片，不能分别解压。

```bash
python restore_v366_handoff.py --bundle-dir /path/to/downloads --destination /path/to/ADMET-v366-restored
```

恢复脚本只用 Python 标准库，先校验源码包、每个分卷及合并包 SHA256，再安全解压，最后逐文件校验；已有不同内容时拒绝覆盖。恢复目录包含当前源码、文档、两阶段输入及控制预测、最终权重、预测、清单、三个基线/候选提交。不要把 GitHub 根目录旧文件覆盖到恢复后的新目录。

源码包补齐历史源码依赖，不代表每个旧脚本都是当前运行入口。不要批量运行历史 finish/finalize 脚本。

六份最终模型包含：v364 脂溶性双种子、v364 溶解度双种子、v366 LD50 双种子。全部开发检查点仍在原远端，GitHub 包含开发预测和验证证据。早期整个历史集成不能仅凭这六份权重对任意新分子完整重建。

## GPU 主机与实际环境

用户指定后续所有 GPU 训练使用：

```bash
ssh -p 24365 root@connect.nmb2.seetacloud.com
```

密码不随仓库交付，由项目持有人提供或使用已有授权连接。

远端 Python `/root/miniconda3/bin/python`；GPU RTX 3090 24GB。实测环境：Python 3.12.14、torch 2.8.0+cu128、numpy 2.5.4、scipy 1.18.1、scikit-learn 1.9.1、RDKit 2026.09.1。旧环境快照不可代替当前环境，见 `docs/v367_remote_weight_manifest.json`。

远端目录 `/root/autodl-tmp/transformer_stage1_v364`、`/root/autodl-tmp/transformer_stage2_v366`；本地 `D:/Projects/ADMET-Competition-Model`，用户原路径通过 junction 指向它。

冻结源码、数据清单和配置共同决定模型签名。必须保留字节，不要批量格式化或改换行。第一阶段签名 `c289b2fc1814c035279448b84726dafed8e6efe53da8f9727bd35f4088d7f2db`，第二阶段 `a906da1890fc76b4f9b353545f2eb3c18fa5226ac5734267ffebbcd8a8598ac6`。

## 实验结果与算法

主干为 6 层、192 维、6 头原子/键 Graph Transformer，约 286 万参数；有图距离与键注意力偏置、描述符分支、rank8 适配器。只在官方拟合结构上做掩码重建和视图一致性预训练，每分区重新训练，排除本轮全部任务查询骨架。双固定种子，从零、自监督、相关任务迁移三条消融，任务独立选头与融合。

第一阶段 v364：六项中只采用脂溶性和溶解度 20% 数值融合。四分区均正、严格验收失败，以实验身份交付。官方两项 MAE 实际下降 0.955%、1.641%，总分由 v361 的 51.240 升至 51.269。Caco2、CYP2D6 抑制、hERG 开发未通过；肝细胞清除率确认失败。

第二阶段 v366：

|任务|配方|fold0 MAE下降|fold1 MAE下降|fold2 MAE下降|valid MAE下降|处理|
|---|---|---:|---:|---:|---:|---|
|PPBR|family_transfer，20%|0.138%|0.153%|-0.006%|0.663%|确认失败，保留原预测|
|LD50|family_transfer，20%|0.708%|0.822%|1.475%|1.073%|实验候选，严格失败|

PPBR 冻结主干训练适配器及头，固定裁剪 [0,100]；LD50 解冻末两层和适配器。两项优化原尺度 L1，拟合集 median/IQR 仿射缩放后最终逆变换。LD50 迁移任务为 AMES、DILI、溶解度，各 120 步；目标拟合最多 1600 步。

LD50 fold2 仅 9 个骨架组，90% bootstrap 区间 [-4.374%, 1.501%]；valid 1930 组，区间 [0.458%, 1.598%]。四分区正向不等于严格通过。融合在 fold0/1 冻结，未根据确认结果另换配方。

## 源码与证据入口

|文件|作用|
|---|---|
|`tools/v364_graph_transformer.py`|图特征、网络、损失、预训练与验证；冻结基础实现|
|`tools/v364_prepare_stage1.py`、`v364_parallel_stage1.py`、`v364_parallel_confirmation.py`、`v364_after_training.py`|第一阶段准备与完成流程；历史已完成，不要覆盖重跑|
|`tools/v365_record_v364_feedback.py`|登记用户官方反馈并保留严格失败记录|
|`tools/v366_graph_transformer.py`|第二阶段固定任务、种子、迁移家族及微调策略|
|`tools/v366_prepare_stage2.py`|准备输入；已存在时拒绝覆盖|
|`tools/v366_run_stage2.py`|开发、唯一配方冻结、确认、最终训练、核验与采集|
|`tools/v366_verify_remote.py`|独立重算选型与 bootstrap、手写注意力、CPU 对照、标签扰动和 GPU 重放|
|`tools/v366_deliver_stage2.py`、`v366_verify_local.py`|融合、CSV、分子顺序及归档核验|
|`docs/platform_scorecard_v364_20261104.json`、`V364_OFFICIAL_FEEDBACK.md`|官方最好成绩证据|
|`docs/V366_STAGE2_RESULTS.md`、`v366_final_quality.json`|第二阶段结论和质量结果|
|`docs/CURRENT_V361_TASK_MODEL_INVENTORY.md`|22 项原模型；加 v364 两项增量和 v366 单项候选即得当前路由|
|`docs/TRANSFORMER_22_TASK_PLAN.md`|22 项计划、计分情景、失败证据|

恢复后只核验当前 CSV 与证据，无训练：

```bash
python tools/v366_verify_local.py delivery
```

完整独立 GPU 核验需要原远端的全部开发检查点。GitHub 六份最终权重包不包含所有开发模型，不应将缺失文件当作核验失败。

确定性 `graphs.pt` 未打包，原远端缓存仍在。新主机可把恢复后的 `experiments/transformer_stage1_v364` 复制到原远端目录，在匹配环境执行 `python -u v364_graph_transformer.py graphs` 重建。第二阶段 wrapper 指向第一阶段绝对路径，恢复到同路径最直接。新阶段必须换签名与缓存目录，不能复用含新查询结构的旧预训练编码器。

## 下一位的顺序

1. 先取得 v366 手动提交后的完整成绩单，核对仅 LD50 改变，再决定是否替代 v364；不要按编号升级最好版本。
2. 保留 v364 回退 CSV 和严格失败证据。官方未涨分时记录真实结果，不反向调同一确认集权重。
3. 优先检验新信号：扩展尚未尝试新图 Transformer 的 AMES、微粒体清除率、VDss，或提出不同于已完成消融的新假设；先设有限候选与预算。
4. 开发选唯一配方后冻结，确认失败不后补换方；明确披露 OOF 代理、公开 valid 长期复用、骨架组不足。
5. 只在指定远端训练。官方页面禁止其他外部数据，新增外部权重也没有明确授权，不直接引入。HIA 官方 AUROC=1，保持输出。

此前 52 分情景是给定多项指标改善后的条件算术，不是概率或成绩预测。单 LD50 若官方 MAE 降低 1%，总分仅约 +0.0114；本轮不能承诺大幅跃升。

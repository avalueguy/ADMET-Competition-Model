# 官方数据版本 v67/v68 复现

训练使用用户授权的 AutoDL A100 40GB，工作目录 `/root/autodl-tmp/admet_v67`。代码中的根目录由源码位置确定，可在其他 Linux CUDA 目录恢复。Python 和依赖的实际版本以 `admet_model_outputs_v67/environment.json` 为准；不需要外部预训练权重。

## 恢复

将 `releases/official_only_v67_v68_20261020.tar.gz` 解压到新目录。先对照 `official_only_v67_v68_archive.json` 检查整包 SHA256，再对照包内 `releases/official_only_v67_v68_manifest.json` 检查每个文件。包内含官方输入、特征缓存、分折、所有开发预测、最终权重和提交；不含 SSH 密码。

不要变更 v67/v68 训练源码的字节、协议或数据后复用原缓存：它们参与签名。新实验应另开版本和输出目录。官方数据只有 train/valid 标签用于训练，test 标签不读取。

## 检验交付结果

在恢复后的项目根目录执行：

```bash
python tools/test_v67_integrity.py
python tools/verify_v67_models.py
python tools/verify_v68_models.py
```

v67 检查需要 CUDA。它重载 8 个最终快照及保存的预处理状态，复算模型输出和最终融合。v68 重载每个接受任务的 4 个结构核模型，重新生成查询核并复算融合；未接受任务逐字保留 v67 行。两个脚本都核对官方模板、哈希和导出结果。格式检查也可单独在本机用标准库执行：

```bash
python tools/verify_submission.py "admet_submissions/v68_official_only/我从山北来带着兰花草-小分子药物预测技术创新赛-20261020.csv"
```

## 重新训练

在独立目录保留源码、协议、官方输入、官方模板及数据清单，不复制已有输出缓存。依次执行：

```bash
python tools/test_v67_integrity.py
python tools/run_v67_pipeline.py
python admet_v68_kernel.py prepare
python tools/run_v68_pipeline.py
python tools/report_v68_valid.py
python tools/package_official_only_release.py
```

`run_v67_pipeline.py` 先构建官方结构特征、烟雾测试，再运行三折开发、额外骨架留出、官方 valid 诊断和 train+valid 最终拟合。v68 在相同划分下评估结构核，冻结选择后导出并独立重载。中断后可重跑，缓存只有签名、拟合行和查询行一致才会使用。用新的提交日期重训练时，应显式调用 v67 导出入口的 `--date YYYYMMDD` 并更新后续记录。

## 指标解释

MAE 越低越好，其增益记录为 `(旧 MAE - 新 MAE) / 旧 MAE`；AUROC、AUPRC、Spearman 越高越好，其增益记录为新值减旧值。不同任务的增益不能直接换算比赛总分。

三折用于模型与权重选择，额外留出也用于门控，因此都属于开发过程；留折诊断不能视为完整嵌套交叉验证。官方 valid 在冻结后仅作诊断，且其中数据在更早项目开发中已被观察。最终泛化提升以真实平台评分为准。

历史 v63 的用户反馈为 50.603，第十名为 51.265。历史 v66 含预训练来源，其资格尚未确认；该复现包仅针对不使用外部样本和预训练权重的 v67/v68。原 v64 缺失门控仍未完成。

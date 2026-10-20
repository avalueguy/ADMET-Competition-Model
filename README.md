# ADMET Competition Model — 2026-10-20 交接

面向小分子 ADMET 竞赛的多任务模型、历史集成基线、训练协议、预测文件与验证证据。**最新完成的模型分支为 v65；v64 是尚未完成的跨分支汇总流程。当前精确官方得分未知，不能将内部验证增益换算为比赛总分。**

本仓库为私有交接项目。请给接手者添加仓库访问权限，再分享此仓库及 Release 链接。

## 从这里开始

1. 阅读 [完整交接日志](docs/HANDOFF.md) 和 [当前状态](CURRENT_STATUS.json)。
2. 获取 [handoff-2026-10-20 Release](https://github.com/avalueguy/ADMET-Competition-Model/releases/tag/handoff-2026-10-20) 的五个 `.tar.gz` 附件；查看 [附件说明](docs/ARTIFACTS.md)。
3. 按 [恢复与复现](docs/REPRODUCE.md) 校验 SHA256、恢复运行目录和环境。
4. 先运行轻量检查，再在获授权的远程 CUDA 主机上运行模型重载检查。没有在本机执行 GPU 训练。

## 版本状态

| 版本 | 方法 | 当前状态 | 提交文件 |
| --- | --- | --- | --- |
| v60 | 既有集成 + 图核/神经分支 | 历史回退基线 | [v60](admet_submissions/v60/) |
| v61 | 频率加权稀疏分子指纹核 + 跨任务图表型核 | 已导出、重载通过；额外化学门控缺失 | [v61](admet_submissions/v61/) |
| v62 | 残差 GINE、虚拟节点、注意力/多层读出、任务门控 | 训练、额外划分、重载、文件复算完成 | [v62](admet_submissions/v62/) |
| v63 | TabICLv2，结构描述符/图和序列特征 | 条件化、额外划分、重载、文件复算完成 | [v63](admet_submissions/v63/) |
| v65 | TabICLv2 + 同分子其他公开任务测量值 | 最新完成分支；条件化验证，非完全未知骨架泛化 | [v65](admet_submissions/v65/) |
| v64 | v61/v62/v63/v65 按任务选择合并 | **未完成，不存在可交付 v64 CSV** | 不应将流程编号视作完成版本 |

所有已导出候选都等待真实平台评测；没有证据证明 v65 的官方得分高于 v60/v61/v62/v63。

## 包含什么

- 根目录的 `admet_*.py` 与协议：保留远程签名所需的原始字节和历史导入依赖。主入口是 `admet_v61_run.py`、`admet_v62_run.py`、`admet_v63_run.py`、`admet_v65_run.py`。
- `admet_model_outputs_v60` 至 `v65`：轻量结果、逐任务验证、模型重放与训练日志。
- `admet_submissions/`：实际 CSV 和独立格式/融合复算结果。
- `docs/`：交接、恢复、实际环境、已知问题与历史说明。
- Release 附件：必要输入与特征、源图模型、最新分支权重/条件化模型、精确远程源码和历史元数据。总计约 1.46 GB。

原始模型代码包含固定远程路径和源代码哈希校验。**不要直接批量修改路径或换行符再加载旧缓存**；恢复时使用说明中的固定目录。新增实验应另开版本/目录。

## 快速轻量校验

在仓库根目录运行（无需 GPU，无需安装训练库）：

```bash
python tools/verify_handoff.py
python admet_release_local_verify.py 65
```

完整推理与训练需要 Release 附件和匹配环境。历史集成的真实基线以 v60 CSV 为准；更早的部分训练权重未归档，不能声称已能对任意新分子完整重建整个历代集成。

# 恢复与复现

## 环境与目录

已核实运行环境：Linux、Python 3.10.12、RTX 4090 24 GB、PyTorch `2.13.0+cu130`、CUDA runtime 13.0、RDKit `2022.09.5`。其余版本见 `environment_snapshot.json` 与 `requirements-reference.txt`。早期项目根目录中的 `admet_*requirements.txt` 与当前环境不一致，不能当作本轮的精确环境锁。

当前源码和持久化 TabICL 模型使用固定路径 `/home/czp123/admet_v23_20260928`。源模型签名包含源码及协议的字节哈希；不要修改源码、协议和数据后继续使用旧缓存。跨 GPU/库版本重放可能存在浮点差异，必须重新通过对应验证脚本。

## 下载与解包

1. 克隆私有仓库；将五个 Release 附件下载到单独目录。
2. 用 `docs/artifact_manifest.json` 或 Release 的 `SHA256SUMS` 核对文件哈希。
3. 在一台新的获授权 Linux 主机上准备上述固定目录，将仓库内容复制到该目录。不要覆盖正在工作的训练目录。
4. 在该目录按以下次序解压（附件只包含项目内相对路径）：

```bash
tar -xzf /path/to/assets/runtime-inputs.tar.gz
tar -xzf /path/to/assets/graph-sources.tar.gz
tar -xzf /path/to/assets/latest-models-v61-v65.tar.gz
tar -xzf /path/to/assets/exact-remote-source-and-metadata.tar.gz
tar -xzf /path/to/assets/fold-state.tar.gz
```

最后一个包保留签名所需的远程源码和历史 JSON。它不应覆盖 README、docs、tools 或 CURRENT_STATUS.json。解包后运行 `python tools/verify_handoff.py --artifacts-dir /path/to/assets --check-restored`，检查所有归档成员的 SHA256。

## Python 依赖

使用 Python 3.10 创建 `.venv`。先按运行主机条件安装匹配的 PyTorch CUDA 构建，再参考 `requirements-reference.txt` 安装数值、化学和建模依赖。参考文件记录实际版本，不保证未来软件源持续提供相同 wheel。

TabICL 在原环境中隔离安装在 `admet_runtime/v63_packages`。恢复时执行：

```bash
.venv/bin/python -m pip install --no-deps --target admet_runtime/v63_packages tabicl==2.2.0 einops==0.8.1
```

包未包含 `.venv`，也没有包含服务器账户或 SSH 密钥。TabICL 公共预训练权重已放在 runtime-inputs 附件中；不要在推理时自动下载不同权重。

## 验证顺序

```bash
python tools/verify_handoff.py
python admet_release_local_verify.py 61
python admet_release_local_verify.py 62
python admet_release_local_verify.py 63
python admet_release_local_verify.py 65
```

接着在远程 CUDA 环境运行（不进行模型拟合）：

```bash
.venv/bin/python admet_v61_verify.py
.venv/bin/python admet_v62_verify.py
.venv/bin/python admet_v63_verify.py
.venv/bin/python admet_v65_verify.py
```

这些脚本重建特征、加载权重并检查索引、签名、预测和最终融合。先完整保存原始 PASS 报告，再在新目录做任何新实验。只加载来自本项目可信交接源的 `.pt/.joblib` 序列化文件。

## 继续本轮未完成的汇总

v64 缺少 `admet_model_outputs_v61/independent_chemical_gate.json`。先在远程运行：

```bash
.venv/bin/python -u admet_v61_extra_gate.py > admet_model_outputs_v61/extra_gate-resume.log 2>&1
```

确认文件 COMPLETE 后，且四个源的 `model_replay.json` 均 PASS，再运行：

```bash
.venv/bin/python -u admet_v64_finish.py
python admet_v64_local_verify.py
```

该补充验证会重新拟合 ExtraTrees 对照，可能耗时较长。不要绕过它、伪造 PASS 或仅修改等待状态。v64 的历史 pipeline_status.json 是超时前遗留状态，不表示进程仍活跃。2026-10-20 核对时没有运行中的 ADMET v61-v65 进程。

## 新实验

不要在现有缓存目录内更改训练协议。开新版本（例如 v66）和新产物目录，先固定方法与验证规则，再训练。把每次实际上传的 CSV SHA256、时间、官方分数和平台是否只显示历史最高分记录在同一张运行表内。没有官方反馈时不要持续以本地代理分数微调并宣称比赛得分上升。

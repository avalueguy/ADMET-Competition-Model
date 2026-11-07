# 激进版 v69/v70 复现

使用官方 train/valid 标签与 SMILES，不使用外部数据或预训练权重；test 标签不解析。训练运行于用户授权的 AutoDL A100 40GB。原 v67/v68 数据、模板、源码和模型签名保持不变。

v69 新增 descriptor/hybrid 两类 GPU CatBoost，在三折全局骨架划分和官方 valid 上早停，以验证最佳树数中位数进行两种子的最终 train+valid 拟合。按任务搜索共享/门控网络、四个结构核与两类树模型的三步融合。目标为 0.65 × 三折平均增益 + 0.35 × valid 增益 − 0.03 × 三折标准差。

v70 加入五类稀疏片段指纹：Morgan 半径 1/2/3、feature Morgan 半径 2、拓扑扭转。一个核使用半径 2 二值指纹，另一个按训练分子中的片段频率加权多视图 log-count Tanimoto；每类核拟合 C=1/10/100。没有额外固定长度折叠，RDKit 片段本身的哈希仍可能碰撞。词表仅来自官方分子结构，频率权重只来自该次拟合分子。与 v69 比较后逐任务自动融合。

开发集同时用于早停、融合和候选选择，开发收益有乐观偏差，不作为独立泛化估计，也不能换算比赛总分。v68 格式已被用户确认通过，但确切官方得分需明天查看；目前仍未进入前十。

## 检查与运行

需先恢复已校验 v67/v68 基础包（SHA256 `ae4dac28be3a399410cae70c1ab7421aceadaaf03eba9b5d6300eda063244e0b`），再恢复对应 v69/v70 增量包。每包外层 record 指定整包 SHA256，manifest 指定逐文件 SHA256。环境版本记录在各输出目录的 environment.json。

```bash
python tools/test_v69_serialization.py
python tools/test_v70_kernels.py
python tools/run_v69_pipeline.py
python tools/report_v69_development.py
python tools/package_v69_release.py
python tools/run_v70_pipeline.py
python tools/report_v70_development.py
python tools/package_v70_release.py
```

v70 的开发核可与 v69 GPU 训练并行；最终选择会等待 v69 完成独立重载校验。保持源码和协议的精确字节后可断点继续；变更源码需单独归档旧输出并重跑，不能改签名绕过缓存验证。

v69 最初一轮因 Spearman target 中的 ndarray 不能写入 JSON 而失败，已在 `v69_failed_serialization_attempt` 保留失败源码、日志和输出。修正版本将数组保存为列表，模型、预处理和 metadata 完成后才发布 predictions 缓存；通过半衰期任务 GPU 回归测试后，旧 v69 缓存未复用，重新训练。v67/v68 缓存只有原签名和拟合/查询行匹配时复用。

最终校验：

```bash
python tools/verify_v69_models.py
python tools/verify_v70_models.py
```

校验会重载每个最终树或核模型，重建预处理或查询核并复算融合。v69 共享/门控神经模型已在基础 v67 交付中独立重载；v70 要求 v69 重载 PASS。没有变化的任务逐字符串保留父版本行。最终再次只向原官网模板插入 y_pred，保持 `5348064.0` 等 Drug_ID 文本、行序、BOM、CRLF；未自动上传平台。

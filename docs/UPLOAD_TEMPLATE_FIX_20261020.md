# 上传格式纠错与平台证据

用户反馈的实际错误为：

```text
submission validation failed: cyp2c9_veith: Drug_ID mismatch at sample_idx 0: expected '5348064.0', received '5348064'
```

因此平台按字符串要求保留 `.0`。以平台实际 expected 为准，使用原官网模板 `admet_official/submission_template.csv`（SHA256 `8ee41589d2c98e7b93081c058b3f928ac09bea4b3ffc899fa13677051e9baa20`）。该模板在报错位置的 ID 为 `5348064.0`。

此前按用户后来提供的 `D:/bili_download/upload_template.csv`（SHA256 `9604b78fd59cb2e7a803df4ed731e054f152af401c4ec6ec79d11af70cfd292b`）改成整数的方向错误，现撤回。两份模板有 7,611 个 ID 字符串差异：CYP2C9 2,419、CYP2D6 2,626、CYP3A4 2,467、DILI 96、溶解度 2、VDSS 1。不能据此推断用户曾用何种软件改写模板。

可上传 v68：`admet_submissions/v68_official_only/我从山北来带着兰花草-小分子药物预测技术创新赛-20261020.csv`。用户方便取用的副本为 `D:/bili_download/v68_平台要求保留小数ID/我从山北来带着兰花草-小分子药物预测技术创新赛-20261020.csv`。二者 SHA256 均为 `0aa975b4e2dfce0890296854fbe621de0beb4955bfd5ed0a0131e88cd4625ccf`。

已逐行检查全部 16,379 行、22 个任务、6 列；所有非预测字段与原官网模板完全一致，全部预测字符串与原 v68 一致。按模板字节插入预测后，反向擦除预测能精确还原原模板，BOM 和 CRLF 保留。没有重新训练或改变预测分数。直接上传 CSV，避免经 Excel/WPS 另存而改变 ID 文本。

整数版本（SHA256 `bc7575519ff011dc0a69cfe31af9e84c5d1bcd23376a4f94bb9fe39f133ef26c`）现标记为勿上传，保留仅作审计。v69 的最终导出及独立校验同样改用原官网模板。原 v67/v68 模型签名、原发布文件和归档均保持原始字节。

用户随后确认修正后的格式通过，当前未进入前十。官方分数需明天才能查看，暂记为空，不推断具体分数。激进版 v69 继续训练。

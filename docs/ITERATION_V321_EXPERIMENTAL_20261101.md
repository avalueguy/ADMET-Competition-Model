# v321 实验 CSV 交付

**未经统计稳定性确认，仅供手动官方测试。** 当前最好官方成绩仍为用户报告的 51.205，v321 尚无官方成绩，不能宣称达到 52。

以 v313 为基线，仅替换 CYP3A4 底物 135 条预测，使用 v315 在开发阶段冻结的秩融合权重 0.20、原两个种子和唯一配置。其余 21 项预测及全部非预测字节精确保留，包含本次已获官方提升的 PPBR 与脂溶性。

四分区 AUROC 改善为 +0.007724、+0.007910、+0.002220、+0.014337，但确认骨架 bootstrap 区间均跨零，v315 严格拒绝状态不变。用户此前明确选择“另做实验 CSV 手动测试”，本交付沿用该实验方式，不扫描新权重，不纳入退步的肝细胞清除率。

最终模型已完成独立特征重建、标签隔离、模型重载、注意力参考和倒序查询检查。本地另行复算秩融合，检查 16,379 行、22 个任务、原 Drug_ID、模板字段、BOM/CRLF、21 项预测字节与所有非预测字节，全部通过。

CSV：`admet_submissions/v321_experimental_cyp3a4/我从山北来带着兰花草-小分子药物预测技术创新赛-20261101.csv`。

CSV SHA256：`e87ba6d09c1bd9e829ecbc74a43aaade023dc442e4f95825d9304a4cfe5f7466`。

归档：`releases/v321_experimental_cyp3a4_20261101.tar.gz`，22 个成员和 50 项输入依赖本地核验通过；SHA256：`fa05874ebdb63baaf1b4f17e34119eef9bd7c1fd86a9a41cc392092d9f5c3348`。

详见 `v321_experimental_delivery.json`、`v321_local_delivery_verification.json`、`v321_local_archive_verification.json`。没有自动上传平台。

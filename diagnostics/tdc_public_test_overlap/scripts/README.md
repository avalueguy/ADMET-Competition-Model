# 诊断脚本

`verify_offline.py` 是便携只读复核入口，以相邻source目录的原始ZIP逐行验证两个CSV、22项指标、总分与哈希。

`original/`保留实际生成文件时的精确源码和原始绝对路径，供审计。JavaScript生成脚本使用`@oai/artifact-tool`创建数值列，再仅替换模板y_pred字段字节。依赖为Codex工作区提供的Artifact Tool运行时，不把node_modules上传。迁移生成流程时需适配原脚本ROOT、REPO与输入路径；无需重新生成即可使用便携复核入口。

全部CSV均为公开标签辅助诊断，不能纳入模型训练或声称模型提分。

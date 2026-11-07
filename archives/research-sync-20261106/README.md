# 当前源码与证据快照

从[本轮Release](https://github.com/avalueguy/ADMET-Competition-Model/releases/tag/research-sync-20261106-offline-diagnostics)下载 `admet-research-source-20261106.zip`。该包保留当前根目录源码、tools、docs、v387已发布快照及experiments轻量记录的原路径。此目录的`source_manifest.json`逐文件记录字节数和SHA256。它与冻结训练输入和权重分开；解压不会自动训练或提交。

最新大文件及诊断集合下载：[本轮Release](https://github.com/avalueguy/ADMET-Competition-Model/releases/tag/research-sync-20261106-offline-diagnostics)。先按当前交接恢复旧运行依赖，再使用本快照更新源码和证据。

快照不包含外部测试标签目录；标签独立保存在diagnostics/，避免混入模型输入。也不包含node_modules、虚拟环境、浏览器截图和密码。

"""Prepare a documented GitHub sync without changing frozen model sources."""
from pathlib import Path
from datetime import datetime, timezone
import ast
import csv
import hashlib
import io
import json
import re
import shutil
import subprocess
import zipfile

ROOT = Path(__file__).resolve().parents[1]
SOURCE = Path(r'C:\Users\lenovo\Documents\ChatGPT\多媒体\tdc_overlap_diagnostic')
DEST = ROOT / 'diagnostics/tdc_public_test_overlap'
PUBLISH = ROOT / 'work/github_sync_20261106'
BASE_URL = 'https://github.com/avalueguy/ADMET-Competition-Model'
TAG = 'research-sync-20261106-offline-diagnostics'


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def copy(source, target):
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        assert sha(source) == sha(target), f'Refusing to replace different artifact: {target}'
    else:
        shutil.copyfile(source, target)
    assert sha(source) == sha(target)


def main():
    PUBLISH.mkdir(parents=True, exist_ok=True)
    old_readme = ROOT / 'docs/history/README_before_repository_sync_20261106.md'
    if not old_readme.exists():
        copy(ROOT / 'README.md', old_readme)
    for name in ('HANDOFF_CURRENT.md', 'CURRENT_STATUS.json', 'docs/GITHUB_UPLOAD_CONTENTS.md'):
        target = ROOT / 'docs/history' / (name.replace('/', '_').replace('.', '_before_sync_20261106.', 1))
        if not target.exists():
            copy(ROOT / name, target)

    # Exact public source bytes and the two label-derived fixtures stay isolated.
    source_files = {
        'tdc_admet_group_public.zip': 'source/tdc_admet_group_public.zip',
        'matched_rows.json': 'public_labels/matched_rows.json',
        'outputs/我从山北来带着兰花草-小分子药物预测技术创新赛-20261106.csv': 'public_labels/PUBLIC_TEST_LABELS_NOT_MODEL.csv',
        'synthetic_offline/OFFLINE_SYNTHETIC_seed20261106.csv': 'synthetic_offline/OFFLINE_SYNTHETIC_seed20261106.csv',
        'synthetic_offline/synthetic_score_report.json': 'synthetic_offline/synthetic_score_report.json',
        'synthetic_offline/independent_verification.json': 'synthetic_offline/independent_verification.json',
        'synthetic_offline/README.md': 'synthetic_offline/README.md',
        'diagnostic_file_verification.json': 'reports/diagnostic_file_verification.json',
        'diagnostic_summary.json': 'reports/diagnostic_summary.json',
        'overlap_report.json': 'reports/overlap_report.json',
        'platform_rejection.json': 'reports/platform_rejection.json',
        'v387_public_label_metric_check.json': 'reports/v387_public_label_metric_check.json',
        'README.md': 'reports/original_overlap_notes.md',
        'prepare_diagnostic.py': 'scripts/original/prepare_diagnostic.py',
        'build_diagnostic.mjs': 'scripts/original/build_diagnostic.mjs',
        'build_synthetic_offline.mjs': 'scripts/original/build_synthetic_offline.mjs',
        'verify_synthetic_offline.py': 'scripts/original/verify_synthetic_offline.py',
    }
    for src, target in source_files.items():
        copy(SOURCE / src, DEST / target)
    copy(ROOT / 'admet_official/submission_template.csv', DEST / 'source/submission_template.csv')
    copy(ROOT / 'docs/platform_scorecard_v387_20261106.json', DEST / 'source/reference_scores_v387.json')
    synthetic = json.loads((DEST / 'synthetic_offline/synthetic_score_report.json').read_text(encoding='utf-8'))
    assert sha(DEST / 'synthetic_offline/OFFLINE_SYNTHETIC_seed20261106.csv') == synthetic['output_sha256']
    assert sha(DEST / 'public_labels/PUBLIC_TEST_LABELS_NOT_MODEL.csv') == 'c9f4839000828c35a0456387dc896d8ea4d4583d15cd748f72390adeb410b58f'
    (DEST / 'README.md').write_text('''# TDC 公开测试标签与离线合成诊断

这里的“标准答案”是公开 TDC ADMET Group 的测试观测值，不是赛事方另外发布的答案。22 个任务、16,379 行的顺序 SMILES 与本地模板一致，保留了重复测量的原始行顺序。

|目录|内容|用途与状态|
|---|---|---|
|public_labels/|公开标签 CSV、逐行来源记录|外部观测查表，非模型预测；此前按此生成的文件已被平台拒绝|
|synthetic_offline/|随机扰动 CSV、22项得分和独立复核|本地合成分 61.775810247；仅检查评分计算|
|source/|原始 TDC ZIP、比赛模板、归一化参考分|固定来源与校验依据|
|reports/|重合证据、拒绝记录、历史指标对照|不作为模型性能证据|
|scripts/|便携复核入口和原始生成脚本|original/ 保存原路径版脚本，不在新机器直接运行|

每项随机选约30%行，共4,917行；其他11,462行保留公开标签。分类选中值替换为0–1均匀随机数，回归任务加入任务尺度随机扰动。固定种子20261106，详细方法见合成得分报告。

61.775810247由同一组已知标签计算，不代表模型准确性、泛化或合规提分。纯标签理论分为65.719642407，平台反馈“22 tasks reach the theoretical optimum; test labels suspected”。没有为随机扰动文件取得平台成绩。

已保存赛事规则禁止外部数据，并要求模型自动生成预测，因此这两个文件都不是合规模型提交。不与训练数据、模型基线混用。本目录不改变v387的用户报告成绩51.314。

来源：https://dataverse.harvard.edu/api/access/datafile/4426004?format=original

源ZIP SHA256：bd0005246cb6f1672333c28ad202112a8bf071ab65d847848715030308faf1f1

v387对照：21项指标与用户分项一致；溶解度本地公开标签MAE为0.7008209745143054，用户报告为0.7006168876998713，关联差异尚未解决。

离线复核：安装Python的numpy、scipy、scikit-learn后，运行 `python diagnostics/tdc_public_test_overlap/scripts/verify_offline.py`。此入口只读取数据与校验，不上传、不训练。生成脚本和Artifact Tool依赖说明见scripts/README.md。
''', encoding='utf-8')
    (DEST / 'scripts/README.md').write_text('''# 诊断脚本

`verify_offline.py` 是便携只读复核入口，以相邻source目录的原始ZIP逐行验证两个CSV、22项指标、总分与哈希。

`original/`保留实际生成文件时的精确源码和原始绝对路径，供审计。JavaScript生成脚本使用`@oai/artifact-tool`创建数值列，再仅替换模板y_pred字段字节。依赖为Codex工作区提供的Artifact Tool运行时，不把node_modules上传。迁移生成流程时需适配原脚本ROOT、REPO与输入路径；无需重新生成即可使用便携复核入口。

全部CSV均为公开标签辅助诊断，不能纳入模型训练或声称模型提分。
''', encoding='utf-8')

    readme = f'''# ADMET 模型研究与交接

更新于2026-11-06（香港时间）。模型最佳用户报告成绩为 **v387：51.314**。最新完成v390没有通过采用门槛，52分及平均MAE降低5%的目标尚未达到。

|材料|成绩或结论|入口|
|---|---|---|
|v387模型预测|用户报告51.314，上传字节未由平台核对|[模型CSV](v387-official-best.csv) · [源码快照](archives/research-sync-20261106/) · [完整交接](HANDOFF_CURRENT.md)|
|v389组合实验|用户报告51.313，已退回v387|[结果说明](docs/V389_OFFICIAL_REGRESSION_AND_V390_PLAN.md)|
|v390辅助残差实验|64次远端GPU种子拟合，无采用任务|[实验结果](docs/V390_RESULTS_AND_HANDOFF.md) · [训练代码](tools/v390_auxiliary_residual.py)|
|61分离线合成文件|本地61.775810247；公开标签加随机扰动，非模型成绩|[合成CSV](diagnostics/tdc_public_test_overlap/synthetic_offline/OFFLINE_SYNTHETIC_seed20261106.csv) · [逐项得分](diagnostics/tdc_public_test_overlap/synthetic_offline/synthetic_score_report.json)|
|公开“标准答案”|TDC公开测试观测查表，纯标签文件已被平台拒绝|[公开标签CSV](diagnostics/tdc_public_test_overlap/public_labels/PUBLIC_TEST_LABELS_NOT_MODEL.csv) · [来源与限制](diagnostics/tdc_public_test_overlap/)|

61.776分是有已知标签参与构造的离线评分测试，不代表模型提升。赛事规则禁止外部数据并要求模型生成预测，这些诊断文件不能作为合规模型提交。模型基线与诊断材料分开保存。

从[当前交接](HANDOFF_CURRENT.md)开始更新模型；查找源码、历史记录和归档请读[仓库目录](docs/REPOSITORY_INDEX.md)。本轮同步范围、校验方式和未打包项见[上传内容说明](docs/GITHUB_UPLOAD_CONTENTS.md)。

本轮所有最新源码、书面证据和轻量实验记录另有[完整源码快照](archives/research-sync-20261106/README.md)。v390完整运行产物在[本轮Release]({BASE_URL}/releases/tag/{TAG})；v380至v389的递归输入与模型包保留在[原交接Release]({BASE_URL}/releases/tag/handoff-v389-20261106)。

保留历史源码的原有路径，避免破坏import、相对路径和冻结签名。训练依赖与旧完整模型限制见交接说明。SSH密码、令牌、本地运行环境与node_modules不纳入同步。
'''
    (ROOT / 'README.md').write_text(readme, encoding='utf-8')
    handoff = f'''# 当前模型与诊断交接

截至2026-11-06（香港时间），模型最佳用户报告成绩：v387=51.314。v389=51.313，回退v387；v390完整实验无采用任务。目标52分和平均MAE降低5%尚未实现。

1. 读取 `docs/V387_RESULTS_AND_HANDOFF.md`、`docs/V389_OFFICIAL_REGRESSION_AND_V390_PLAN.md`、`docs/V390_RESULTS_AND_HANDOFF.md`。
2. v387模型CSV为 `v387-official-best.csv`，SHA256 `97cdbd749fde3ea10c2e55cfea143cfc2e0208b18edc09ebd101feb3d72a77a3`。完整成绩是用户归属，平台未提供上传文件字节哈希。
3. 从[原交接Release]({BASE_URL}/releases/tag/handoff-v389-20261106)下载source、manifest、restore脚本及五份运行归档，按原manifest核对哈希；执行 `python restore_v389_handoff.py --bundle-dir . --destination ../ADMET-restored`。顺序为v380→v382→v385→v388→v389。
4. 将[当前源码快照](archives/research-sync-20261106/README.md)解压到另一个目录核对，再覆盖恢复目录中的源码与文档。快照不会声称覆盖所有早期缺失权重；未重训强历史集成，部分OOF仍是proxy。
5. 若复核v390，从[本轮Release]({BASE_URL}/releases/tag/{TAG})取 `v390_complete_results.tar.gz`，按 `docs/v390_archive_verification.json` 核对。120个不可变输出通过，verification.log追加PASS的活动日志例外已有记录。
6. 后续GPU拟合仅使用用户指定 `connect.nmb2.seetacloud.com:25870`；凭据通过运行时安全输入，不写入仓库。优先恢复历史组件的真实骨架OOF，再验证Caco2条件信息和新表示。

`diagnostics/tdc_public_test_overlap/`与训练分离。这里包含公开TDC测试观测、完全标签查表CSV（平台拒绝）和61.775810247分合成CSV（本地评分、未上传）。用户称“标准答案”的文件是外部公开观测，不是合规模型输出，不能加入比赛训练或提交。

公开标签复算v387时21项一致；溶解度本地MAE0.7008209745143054、用户上报0.7006168876998713，差异尚未解决。保留原CSV字节，不通过改动标签或噪声伪造模型涨分。

最新状态看 `CURRENT_STATUS.json`，历史首页已保存至 `docs/history/`，历史实验报告保持原有时间与结论。
'''
    (ROOT / 'HANDOFF_CURRENT.md').write_text(handoff, encoding='utf-8')
    (ROOT / 'docs/HANDOFF_CURRENT.md').write_text('当前入口已统一为[根目录交接说明](../HANDOFF_CURRENT.md)。旧版记录保留在[历史目录](history/)。\n', encoding='utf-8')
    state_path = ROOT / 'CURRENT_STATUS.json'
    state = json.loads(state_path.read_text(encoding='utf-8'))
    state.update(
        handoff_date='2026-11-06', timezone='Asia/Hong_Kong', known_best_version='v387', official_score=51.314,
        latest_completed_model_branch='v390', aggregate_status='V390_COMPLETE_NO_ACCEPTED_CANDIDATE',
        active_experiment=None, active_experiment_status='V390_COMPLETE_NO_ACCEPTED_CANDIDATE',
        recommended_submission_version='v387', new_submission_created=False, iteration_report='docs/V390_RESULTS_AND_HANDOFF.md',
        current_followup='docs/V390_RESULTS_AND_HANDOFF.md', current_handoff='HANDOFF_CURRENT.md',
        current_experiment_progress='docs/V390_RESULTS_AND_HANDOFF.md', latest_feedback_version='v389',
        last_user_reported_official_version='v389', last_user_reported_official_score=51.313,
        latest_candidate_official_score=51.313, target_achieved=False, active_gpu_training_at_snapshot=False,
        latest_training_observation='v390 complete: 64 remote seed fits; no accepted task; v387 remains best user-reported model.',
        current_state_checked_at_utc=datetime.now(timezone.utc).isoformat(),
        offline_diagnostics=dict(directory='diagnostics/tdc_public_test_overlap',
            synthetic_local_score=synthetic['synthetic_local_score'], synthetic_is_model_result=False,
            synthetic_platform_score=None, public_label_file_platform_status='REJECTED_USER_REPORTED',
            competition_compliant=False, source_type='PUBLIC_TDC_TEST_OBSERVATIONS', excluded_from_training=True),
        public_label_model_comparison_note='21 tasks reproduce v387; solubility local/file-score association discrepancy remains unresolved.',
    )
    save(state_path, state)
    (ROOT / 'docs/REPOSITORY_INDEX.md').write_text('''# 仓库目录

|位置|内容与使用方法|
|---|---|
|README.md / HANDOFF_CURRENT.md|当前状态和接手步骤，优先阅读|
|CURRENT_STATUS.json|当前模型状态；历史releases条目是当时快照|
|tools/|版本化训练、验证、交付脚本，原路径保留|
|docs/|计划、成绩单、失败结果、哈希、交付证据|
|docs/history/|整理前首页与状态，保留历史表述|
|diagnostics/tdc_public_test_overlap/|外部公开标签与61.776分合成数据，严格分开标注|
|archives/research-sync-20261106/|当前全部源码和轻量证据快照，附逐文件manifest|
|源码快照内github_v387_code/|此前发布的v387冻结代码包，原始内容与哈希不改|
|v387-official-best.csv|模型最佳用户报告51.314的原始预测文件|
|v389-experimental.csv|已被后续反馈否定的历史实验，51.313|
|experiments/|本地固定输入、权重和运行证据，体积较大；源码快照含轻量证据，完整数据在Release|
|根目录admet_v*.py|历史训练入口；仍被旧脚本引用，因此不随意移动|

本轮Release提供v390完整运行归档、源码快照、诊断打包与校验清单。以前的Release和根目录v366分卷、v389包保留，避免旧链接失效。旧源码与数据的完整复现限制见当前交接说明。

目录整理由统一入口、用途分类、来源说明和manifest组成，不改写历史实验的冻结脚本或签名。node_modules、虚拟环境、缓存、临时探测响应、浏览器截图及凭据不上传。
''', encoding='utf-8')
    (ROOT / 'docs/GITHUB_UPLOAD_CONTENTS.md').write_text(f'''# 2026-11-06 GitHub同步内容

目标仓库：{BASE_URL}。本次由用户明确要求同步最新内容，包括公开标签及61分离线合成文件。

- 主分支入口更新：README、当前交接、目录索引、状态文件、v390结果与最新训练/验证代码。
- 当前源码快照：根目录历史源码、tools和docs的最新文本、v387已发布包、experiments的轻量JSON/Python记录。保留原路径、逐文件SHA256；不把历史快照当作新的实验结论。
- 公开标签隔离目录：原始公开TDC ZIP、标签CSV、matched_rows、比赛模板、得分参考、逐行重合和平台拒绝证据。
- 合成诊断：固定种子20261106，4,917行扰动，本地61.775810247，原CSV字节与报告完整保留。非模型成绩、非合规参赛文件、没有平台成绩。
- v390完整运行产物：`v390_complete_results.tar.gz`，SHA256 `f765c95280a61dc408592860e7ef06b3aa03807cafddff5b5dd0a3ce2e2c1f80`，68,706,342字节，在[本轮Release]({BASE_URL}/releases/tag/{TAG})。
- v380→v389所需输入/模型包继续从[现有Release]({BASE_URL}/releases/tag/handoff-v389-20261106)获取，不重复改名或改写。

不上传：SSH密码、访问令牌、Cookie、虚拟环境、node_modules、浏览器截图、临时下载探测副本及未独立收集的远端候选权重。不移动被模型脚本引用的历史源码。大文件通过Release，快照通过archives目录交接。

v387模型最佳仍为用户报告51.314；52分及MAE平均降低5%尚未实现。公开标签与随机扰动材料明确标为诊断，并记录赛事规则限制，避免被接手者误作模型提分。
''', encoding='utf-8')
    (ROOT / 'experiments/README.md').write_text('实验运行目录较大，不直接纳入源码Git追踪。轻量JSON/Python证据在当前源码快照内的experiments/原路径；完整输入、权重和运行产物按HANDOFF_CURRENT.md从Release恢复。\n', encoding='utf-8')
    archives_dir = ROOT / 'archives/research-sync-20261106'
    archives_dir.mkdir(parents=True, exist_ok=True)
    (archives_dir / 'README.md').write_text(f'''# 当前源码与证据快照

从[本轮Release]({BASE_URL}/releases/tag/{TAG})下载 `admet-research-source-20261106.zip`。该包保留当前根目录源码、tools、docs、v387已发布快照及experiments轻量记录的原路径。此目录的`source_manifest.json`逐文件记录字节数和SHA256。它与冻结训练输入和权重分开；解压不会自动训练或提交。

最新大文件及诊断集合下载：[本轮Release]({BASE_URL}/releases/tag/{TAG})。先按当前交接恢复旧运行依赖，再使用本快照更新源码和证据。

快照不包含外部测试标签目录；标签独立保存在diagnostics/，避免混入模型输入。也不包含node_modules、虚拟环境、浏览器截图和密码。
''', encoding='utf-8')
    # A portable source snapshot includes all new code/evidence, without vendored dependencies.
    sources = set()
    suffixes = {'.py','.json','.md','.txt','.toml','.yml','.yaml','.mjs'}
    for path in ROOT.iterdir():
        if path.is_file() and path.suffix.lower() in suffixes and not path.name.startswith(('tmp_', 'v118_original_manifest')):
            sources.add(path)
    sources.update(ROOT / name for name in ('.gitignore','.gitattributes'))
    for folder in ('tools','docs','experiments','github_v387_code'):
        for path in (ROOT / folder).rglob('*'):
            if not path.is_file() or any(part in {'node_modules','__pycache__','browser_proof','archive_conflicts'} for part in path.relative_to(ROOT).parts):
                continue
            if path.suffix.lower() in suffixes or folder == 'github_v387_code':
                sources.add(path)
    forbidden = re.compile(rb'ghp_[A-Za-z0-9]{25,}|github_pat_[A-Za-z0-9_]{20,}|-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----|(?i:password|passwd|ssh_pass|access_token)\s*[:=]\s*["\x27][A-Za-z0-9_+/.=-]{8,}["\x27]')
    syntax_errors = []
    entries = []
    for path in sorted(sources):
        content = path.read_bytes()
        assert not forbidden.search(content), 'Potential credential: ' + path.relative_to(ROOT).as_posix()
        if path.suffix=='.py':
            try:
                ast.parse(content.decode('utf-8-sig'))
            except SyntaxError as error:
                syntax_errors.append(dict(path=path.relative_to(ROOT).as_posix(), line=error.lineno, reason=error.msg))
        entries.append(dict(path=path.relative_to(ROOT).as_posix(),bytes=len(content),sha256=hashlib.sha256(content).hexdigest()))
    source_zip = archives_dir / 'admet-research-source-20261106.zip'
    with zipfile.ZipFile(source_zip, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in sorted(sources):
            archive.write(path, arcname=path.relative_to(ROOT).as_posix())
    with zipfile.ZipFile(source_zip) as archive:
        assert archive.testzip() is None
        for entry in entries:
            assert hashlib.sha256(archive.read(entry['path'])).hexdigest()==entry['sha256']
    save(archives_dir / 'source_manifest.json', dict(
        archive=source_zip.name,sha256=sha(source_zip),bytes=source_zip.stat().st_size,
        source_files=len(entries),credentials_pattern_hits=0,files=entries,
        syntax_audit=dict(errors=syntax_errors,note='Historical frozen scripts retained verbatim; this audit does not train models.'),
    ))
    diagnostic_zip = PUBLISH / 'tdc-public-labels-and-offline-synthetic-20261106.zip'
    diagnostic_paths = sorted(path for path in DEST.rglob('*') if path.is_file() and path.name!='artifact_manifest.json')
    diagnostic_manifest = dict(data_type='DISCLOSED_PUBLIC_TEST_LABELS_AND_OFFLINE_SYNTHETIC_NOT_MODEL',
        competition_compliant=False,synthetic_local_score=synthetic['synthetic_local_score'],
        files=[dict(path=path.relative_to(ROOT).as_posix(),bytes=path.stat().st_size,sha256=sha(path)) for path in diagnostic_paths])
    save(DEST / 'artifact_manifest.json', diagnostic_manifest)
    with zipfile.ZipFile(diagnostic_zip,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as archive:
        for path in diagnostic_paths + [DEST / 'artifact_manifest.json']:
            archive.write(path,arcname=path.relative_to(ROOT).as_posix())
    assets = [source_zip, archives_dir / 'source_manifest.json', diagnostic_zip, ROOT / 'releases/v390_complete_results.tar.gz']
    save(PUBLISH / 'release_assets.json', [dict(path=str(path),name=path.name,sha256=sha(path),bytes=path.stat().st_size) for path in assets])
    print(json.dumps(dict(source_files=len(entries),source_zip_bytes=source_zip.stat().st_size,
        source_syntax_errors=syntax_errors,diagnostic_files=len(diagnostic_paths),diagnostic_zip_bytes=diagnostic_zip.stat().st_size,
        release_assets=[str(path) for path in assets]),ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()

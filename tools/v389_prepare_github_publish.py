"""Package the current v389 handoff for browser-authenticated GitHub publishing."""
from pathlib import Path
from datetime import datetime, timezone
import ast
import hashlib
import json
import re
import shutil
import zipfile

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / 'work/github_handoff_v389_20261106'
TAG = 'handoff-v389-20261106'
REPO = 'https://github.com/avalueguy/ADMET-Competition-Model'


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def main():
    DEST.mkdir(parents=True, exist_ok=True)
    load = lambda p: json.loads((ROOT / p).read_text(encoding='utf-8-sig'))
    release = load('docs/v389_release.json')
    deps = load('docs/v389_handoff_dependencies.json')
    delivery = load('docs/v389_delivery.json')
    current = load('CURRENT_STATUS.json')
    current.update(known_best_version='v387', best_official_version='v387',
        best_official_version_user_reported='v387', best_official_score_user_reported=51.314,
        official_best_version_user_reported='v387', official_best_record='docs/current_best_after_v387_20261106.json',
        last_user_reported_official_version='v387', last_user_reported_official_score=51.314,
        official_scorecard_provenance_note='User supplied full v386/v387 scorecards and explicit version association; uploaded byte hash unavailable.',
        active_experiment=None, active_experiment_status='V389_COMPLETE_EXPERIMENTAL_CSV_DELIVERED',
        recommended_submission_version='v387', new_submission_created=True,
        iteration_report='docs/V389_RESULTS_AND_HANDOFF.md', current_followup='docs/V389_RESULTS_AND_HANDOFF.md',
        priority_next_step='Improve molecular representations and scaffold generalization; audit real OOF for the historical ensemble.',
        latest_training_observation='v389 completed remotely. PPBR experimental update only; strict confidence interval gate not passed.',
        last_updated_hong_kong='2026-11-06', current_state_checked_at_utc=datetime.now(timezone.utc).isoformat())
    save(ROOT / 'CURRENT_STATUS.json', current)
    archives = deps['recursive_dependencies']['required_archives'] + [deps['required_archive'], release]
    inputs = []
    for item in archives:
        path = ROOT / item['archive']
        assert path.stat().st_size == item['bytes'] and sha(path) == item['sha256']
        inputs.append(dict(name=path.name, destination=item['archive'], bytes=path.stat().st_size, sha256=item['sha256']))
    csvs = [('v387-official-best.csv', delivery['baseline']['csv'], delivery['baseline']['csv_sha256']),
            ('v389-experimental.csv', delivery['csv'], delivery['csv_sha256'])]
    for name, original, expected in csvs:
        assert sha(ROOT / original) == expected
        shutil.copyfile(ROOT / original, DEST / name)
    report = (ROOT / 'docs/V389_RESULTS_AND_HANDOFF.md').read_text(encoding='utf-8')
    intro = f'''# ADMET 当前交接：v387 官方 51.314 / v389 实验

截至 2026-11-06，官方最好是 **v387：51.314**。v389 尚未取得官方成绩，仅修改 PPBR，四个本地分区 MAE 降低0.13%–0.66%，确认置信区间跨零。平均 MAE 降低5%的目标尚未达到。

下载 [本轮完整交接 Release]({REPO}/releases/tag/{TAG}) 的全部资产，或使用根目录的源码包和两个 CSV。**GitHub 自动生成的 Source code 归档只对应发布时主分支，不能代替本轮手工上传的源码和模型资产。**

1. 安装 Python 3.10+；恢复脚本只用标准库。
2. 下载 `admet-v389-source.zip`、`handoff-manifest-v389.json`、`restore_v389_handoff.py` 和5份手工上传的 `.tar.gz`，放在同一目录。
3. 执行 `python restore_v389_handoff.py --bundle-dir . --destination ../ADMET-v389-restored`。
4. 在恢复目录先读 `docs/V389_RESULTS_AND_HANDOFF.md`、`docs/v389_delivery.json` 和 `docs/v388_dependency_audit.json`，按原脚本与锁定依赖在指定远端继续训练。SSH 密码不在交接内容中。

模型和输入按 v380 → v382 → v385 → v388 → v389 恢复；源码包最后覆盖为当前源码。恢复程序先核验压缩包 SHA256，再检查解包路径，拒绝链接和路径越界。

当前源码、训练脚本、历史书面证据在 `admet-v389-source.zip`；模型、固定输入、预测和分区证据在5份运行归档。历史集成部分 OOF 是代理、早期模型不完整等限制继续保留，不能声称对任意新分子完整重建整个历史集成。

|提交|状态|SHA256|
|---|---|---|
|`v387-official-best.csv`|用户反馈官方51.314|`{delivery['baseline']['csv_sha256']}`|
|`v389-experimental.csv`|尚无官方成绩|`{delivery['csv_sha256']}`|

CSV 在比赛上传时应按平台要求恢复原文件名，见清单中的 `original_path`。不自动提交比赛。

以下为本轮实验结果；更早版本的说明属于历史记录。

'''
    (DEST / 'README.md').write_text(intro + report, encoding='utf-8')
    (DEST / 'HANDOFF_CURRENT.md').write_text(intro + report, encoding='utf-8')
    shutil.copyfile(ROOT / 'CURRENT_STATUS.json', DEST / 'CURRENT_STATUS.json')
    (ROOT / 'docs/GITHUB_UPLOAD_CONTENTS.md').write_text(intro, encoding='utf-8')
    restore = '''"""Verify and restore the five-archive v389 handoff without external packages."""
from pathlib import Path, PurePosixPath
import argparse, hashlib, json, shutil, tarfile, zipfile
def sha(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
 return h.hexdigest()
def safe(root,name):
 p=PurePosixPath(name)
 if p.is_absolute() or '..' in p.parts or '\\\\' in name or ':' in name:raise ValueError('Unsafe archive path: '+name)
 dest=(root/name).resolve()
 if not dest.is_relative_to(root.resolve()):raise ValueError('Path escapes destination')
 return dest
def main():
 a=argparse.ArgumentParser();a.add_argument('--bundle-dir',type=Path,default=Path('.'));a.add_argument('--destination',type=Path,required=True);o=a.parse_args()
 manifest=json.loads((o.bundle_dir/'handoff-manifest-v389.json').read_text(encoding='utf-8'))
 if o.destination.exists() and any(o.destination.iterdir()):raise SystemExit('Destination must be absent or empty')
 items=manifest['runtime_archives']+[manifest['source_archive']]
 for item in items:
  src=o.bundle_dir/item['name']
  if src.stat().st_size!=item['bytes'] or sha(src)!=item['sha256']:raise SystemExit('Checksum mismatch: '+item['name'])
 o.destination.mkdir(parents=True,exist_ok=True)
 for item in manifest['runtime_archives']:
  src=o.bundle_dir/item['name'];copy=safe(o.destination,item['destination']);copy.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(src,copy)
  with tarfile.open(src,'r:gz') as tf:
   for m in tf:
    dest=safe(o.destination,m.name)
    if m.isdir():dest.mkdir(parents=True,exist_ok=True);continue
    if not m.isfile():raise ValueError('Only regular files supported: '+m.name)
    dest.parent.mkdir(parents=True,exist_ok=True)
    with tf.extractfile(m) as r,dest.open('wb') as w:shutil.copyfileobj(r,w)
 with zipfile.ZipFile(o.bundle_dir/manifest['source_archive']['name']) as zf:
  for m in zf.infolist():
   dest=safe(o.destination,m.filename)
   if m.is_dir():dest.mkdir(parents=True,exist_ok=True);continue
   dest.parent.mkdir(parents=True,exist_ok=True)
   with zf.open(m) as r,dest.open('wb') as w:shutil.copyfileobj(r,w)
 for item in manifest['source_files']:
  if sha(o.destination/item['path'])!=item['sha256']:raise SystemExit('Restored source mismatch: '+item['path'])
 for item in manifest['submissions']:
  src=o.bundle_dir/item['name']
  if sha(src)!=item['sha256']:raise SystemExit('CSV checksum mismatch')
  dest=safe(o.destination,item['original_path']);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(src,dest)
 print('PASS: restored source, inputs, model artifacts and submissions; read docs/V389_RESULTS_AND_HANDOFF.md')
if __name__=='__main__':main()
'''
    (DEST / 'restore_v389_handoff.py').write_text(restore, encoding='utf-8')
    ast.parse(restore)
    sources = {p for p in ROOT.iterdir() if p.is_file() and p.suffix.lower() in ('.py', '.json', '.md', '.txt', '.toml', '.yml', '.yaml')}
    for folder in ('tools', 'docs'):
        sources |= {p for p in (ROOT / folder).rglob('*') if p.is_file() and p.suffix.lower() in ('.py', '.json', '.md', '.txt', '.toml', '.yml', '.yaml', '.mjs')
                    and not any(x in p.parts for x in ('browser_proof', '__pycache__', 'archive_conflicts'))}
    sources |= {p for p in (ROOT / '.gitignore', ROOT / '.gitattributes') if p.exists()}
    sources = {p for p in sources if not p.name.startswith(('v367_upload_', 'github-publish-', 'platform_history_edge'))}
    forbidden = re.compile(rb'ghp_[A-Za-z0-9]{25,}|github_pat_[A-Za-z0-9_]{20,}|-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----|(?i:password|passwd|ssh_pass|access_token)\s*[:=]\s*[\"\x27][A-Za-z0-9_+/.=-]{8,}[\"\x27]')
    for p in sources:
        content = p.read_bytes()
        assert not forbidden.search(content), 'Potential credential: ' + p.relative_to(ROOT).as_posix()
    source = DEST / 'admet-v389-source.zip'
    with zipfile.ZipFile(source, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
        for p in sorted(sources):
            zf.write(p, arcname=p.relative_to(ROOT).as_posix())
    with zipfile.ZipFile(source) as zf:
        assert zf.testzip() is None
        for p in sources:
            assert hashlib.sha256(zf.read(p.relative_to(ROOT).as_posix())).hexdigest() == sha(p)
    manifest = dict(version='v389', created_at_utc=datetime.now(timezone.utc).isoformat(), repository=REPO, release_tag=TAG,
        official_best=dict(version='v387', score=51.314), experimental_candidate=dict(version='v389', official_score=None),
        source_archive=dict(name=source.name, sha256=sha(source), bytes=source.stat().st_size), runtime_archives=inputs,
        source_files=[dict(path=p.relative_to(ROOT).as_posix(), sha256=sha(p), bytes=p.stat().st_size) for p in sorted(sources)],
        submissions=[dict(name=n, original_path=p, sha256=h, bytes=(DEST/n).stat().st_size) for n,p,h in csvs],
        credentials_included=False, full_historical_ensemble_reconstructed=False)
    save(DEST / 'handoff-manifest-v389.json', manifest)
    assets = [str(ROOT / r['destination']) for r in inputs] + [str(DEST / name) for name in ('admet-v389-source.zip','handoff-manifest-v389.json','restore_v389_handoff.py','v387-official-best.csv','v389-experimental.csv')]
    save(DEST / 'upload-assets.json', assets)
    print(json.dumps(dict(status='PASS', directory=str(DEST), source_files=len(sources), source_bytes=source.stat().st_size,
        runtime_bytes=sum(x['bytes'] for x in inputs), assets=assets), ensure_ascii=False))


if __name__ == '__main__':
    main()

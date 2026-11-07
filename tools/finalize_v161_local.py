"""Independently reconstruct new task blends and deliver the exact model CSV."""
import json,shutil,sys
from collections import Counter
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'tools'))
from finalize_v124_local import load,save,sha,rows,array,ranks
from verify_submission import verify
from repack_upload_template import field_spans

def main():
    report=load(ROOT/'admet_model_outputs_v161/evidence.json');audit=load(ROOT/'admet_model_outputs_v161/independent_verification.json')
    backup=load(ROOT/'releases/v161_new_structure_models_20261027.archive.json');local=load(ROOT/'docs/v161_local_archive_verification.json')
    assert audit['status']=='PASS' and local['all_member_hashes_verified'] and local['dependency_hashes_verified'] and backup['sha256']==local['archive_sha256']
    anchor=next((ROOT/'admet_submissions/v145_feedback_repair').glob('*.csv'));file=ROOT/report['file'];assert sha(anchor)==report['baseline_sha256'] and sha(file)==report['sha256']
    old,new=rows(anchor),rows(file);assert len(old)==len(new)==16380
    for task,ch in report['selected'].items():
        number=ch['version'];r=ch['record'];cfg=r['config'];folder=ROOT/f'admet_model_outputs_{number}/models/{task}/final'
        if number in ('v150','v157'):path=folder/(cfg['view']+'_C'+str(cfg['C'])+'.npz')
        elif number=='v159':path=folder/('aux'+str(cfg['auxiliary']))/'predictions.npz'
        elif number=='v149':path=folder/('depth'+str(cfg['depth']))/'predictions.npz'
        else:path=folder/('depth'+str(cfg['depth']))/'predictions.npz'
        p=array(path,'p');pos=[i for i,z in enumerate(old) if i and z[0]==task];a=[float(old[i][4]) for i in pos];w=r['weight'];assert len(p)==len(pos)
        value=[(1-w)*x+w*y for x,y in zip(a,p)] if r['metric']=='MAE' else [(1-w)*x/len(a)+w*y/len(a) for x,y in zip(ranks(a),ranks(p))]
        assert all(abs(float(new[i][4])-z)<1e-14 for i,z in zip(pos,value))
    changes=Counter()
    for a,z in zip(old[1:],new[1:]):
        assert a[:4]+a[5:]==z[:4]+z[5:]
        if a[0] not in report['selected']:assert a==z
        if a[4]!=z[4]:changes[a[0]]+=1
    assert dict(changes)==report['verification']['changed_tasks'] and 'hia_hou' not in changes
    artifact=load(ROOT/'work/v161_export/v161_rows.json');assert artifact==new
    template=ROOT/'admet_official/submission_template.csv';fmt=verify(file,template);raw,ref=file.read_bytes(),template.read_bytes();ss,tt=field_spans(raw),field_spans(ref);pieces=[];cursor=0
    for i in range(1,len(new)):
        lo,hi=ss[i][4];a,z=tt[i][4];pieces.extend((raw[cursor:lo],ref[a:z]));cursor=hi
    pieces.append(raw[cursor:]);assert b''.join(pieces)==ref
    assert fmt['rows']==16379 and fmt['tasks']==22 and fmt['utf8_bom'] and fmt['crlf_only']
    fmt.update(nonprediction_bytes_exact=True,independent_final_fusion_replay=True,unchanged_task_predictions_exact=True,artifact_string_rows_verified=True)
    save(ROOT/'docs/v161_local_submission_verification.json',fmt)
    lines=['# v161：新结构表示和多任务模型实验','',
        '基线 v145 官方成绩 51.166；本候选尚无官方成绩，52 分目标尚未达成。','',
        '六条路线分别为目标任务合规近邻特征、展开路径联合核、展开路径稀疏提升树、训练折内路径低秩表示、局部几何与结构分布加权核、纯结构多任务迁移神经网络。每条路线覆盖 21 项，HIA AUROC 已为 1，保留。','',
        '先用 fold0/1 冻结配置与 .2/.45 融合比例，要求全部四个分区的点增益为正、平均达到实际阈值，且 fold2 和 valid 骨架配对重采样 90% 区间下界分别为正。跨路线按最小开发增益选胜者；胜者确认失败则不替换。','',
        '这些标签多次复用，部分历史基线 OOF 是近似对照。上述检查不能消除选择偏差，也不能保证官方提分。v150 确认结果在扩展后续路线前已观察，未放宽接纳门槛。','',
        '| 任务 | 路线 | 融合比例 | 分区 | 基线指标 | 候选指标 | 增益 |','|---|---|---:|---|---:|---:|---:|']
    for task,ch in report['selected'].items():
        for r in ch['record']['confirmation']:lines.append(f"| {task} | {ch['version']} | {ch['record']['weight']:.2f} | {r['part']} | {r['baseline']:.6f} | {r['candidate']:.6f} | {r['gain']:+.6f} |")
    lines+=['','MAE 增益为相对误差减少；其余为指标绝对增益。','']
    for r in report['experiments']:lines.append(f"- {r['version']} 独立路线通过：{', '.join(r['accepted']) or '无'}。最终合入还取决于开发折胜者规则。")
    lines+=['',f"最终改变 {len(changes)} 项、{sum(changes.values())} 条预测；其余 {22-len(changes)} 项的预测字符串与 v145 一致。",'',
        '新模型只使用官方结构与允许的 train/valid 标签。独立模型重载、查询标签扰动、特征重算与融合复算通过；Drug_ID 原字符串、行序及全部非预测字节保持官方模板。没有修改单个测试样本答案。','',
        'v158 因描述符单精度尺度溢出弃用；原部分文件与日志保留，v159 改为双精度统计后重新训练。v153/v159 使用互不重叠的任务分片调度，训练源和模型选择规则不变。','',
        f"CSV SHA256：`{report['sha256']}`",'',
        f"备份：`{backup['path']}`，{backup['files']} 个文件及依赖包哈希已核对。",'',
        '用户已授权通过 Edge 自动提交并读取官方成绩；本文件的提交状态以独立平台回执记录为准。未推送 GitHub，未关闭服务器。']
    doc=ROOT/'docs/ITERATION_V161_20261027.md';doc.write_text('\n'.join(lines)+'\n',encoding='utf-8')
    dest=Path('D:/bili_download/v161_new_structure_models');dest.mkdir(parents=True,exist_ok=True);upload=dest/file.name
    if upload.exists():assert upload.read_bytes()==raw
    else:shutil.copy2(file,upload)
    assert sha(upload)==report['sha256'];shutil.copy2(file.with_suffix('.verification.json'),upload.with_suffix('.verification.json'));shutil.copy2(doc,dest/'v161_说明.md')
    delivery=dict(status='LOCAL_DELIVERY_COMPLETE_READY_FOR_EDGE',version='v161',baseline='v145',baseline_official_score=51.166,csv=report['file'],upload_copy=str(upload),sha256=report['sha256'],changed_tasks=dict(changes),changed_predictions=sum(changes.values()),unchanged_tasks=22-len(changes),independent_final_fusion_replay=True,nonprediction_bytes_exact=True,archive=backup,all_archive_member_hashes_verified=True,dependencies_verified=True,official_score=None,target_achieved=False,target_score=52,automatically_uploaded=False,github_push=False,server_shutdown=False)
    save(ROOT/'docs/v161_local_delivery.json',delivery);save(dest/'v161_delivery.json',delivery)
    p=ROOT/'CURRENT_STATUS.json';state=load(p);state.update(latest_completed_model_branch='v161',latest_completed_model_branch_is_proven_best=False,active_experiment='v161',active_experiment_status=delivery['status'],recommended_submission_version='v161',new_submission_created=True,running_new_training_processes=0)
    state['active_iteration']=dict(version='v161',baseline='v145',status=delivery['status'],official_score=None,running_training_processes=0);state['releases']['161']=delivery;state['v161_result']=delivery;save(p,state)
    print(json.dumps(delivery,ensure_ascii=True,indent=2))
if __name__=='__main__':main()

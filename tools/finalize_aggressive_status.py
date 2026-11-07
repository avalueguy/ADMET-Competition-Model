"""Finalize verified local releases, convenient upload copies and development summaries."""
import hashlib
import json
from pathlib import Path

from verify_submission import verify

ROOT=Path(__file__).resolve().parents[1]


def read(path):
    return json.loads((ROOT/path).read_text(encoding="utf-8"))


def save(path,value):
    (ROOT/path).write_bytes((json.dumps(value,ensure_ascii=False,indent=2)+"\n").encode("utf-8"))


def main():
    status=read("CURRENT_STATUS.json")
    deliveries={}
    for version in (69,70):
        folder=f"admet_model_outputs_v{version}"
        upload=read(folder+"/upload_release.json")
        replay=read(folder+"/model_replay.json")
        pipeline=read(folder+"/pipeline_status.json")
        development=read(folder+"/development_metrics.json")
        archive=read(f"releases/aggressive_v{version}_archive.json")
        restored=read(f"releases/aggressive_v{version}_20261020.tar.restore.json")
        assert pipeline["status"]=="COMPLETE" and replay["status"]==restored["status"]=="PASS"
        assert restored["archive_sha256"]==archive["sha256"]
        file=ROOT/upload["file"]
        audit=verify(file,ROOT/"admet_official/submission_template.csv")
        assert audit["sha256"]==upload["sha256"]==replay["upload_sha256"]
        assert development["independent_generalization_estimate"] is False
        dest=Path(f"D:/bili_download/v{version}_激进迭代_可直接上传")/file.name
        dest.parent.mkdir(parents=True,exist_ok=True)
        if dest.exists():assert dest.read_bytes()==file.read_bytes()
        else:dest.write_bytes(file.read_bytes())
        assert hashlib.sha256(dest.read_bytes()).hexdigest()==audit["sha256"]
        record=dict(csv=upload["file"],local_upload_file=dest.as_posix(),sha256=audit["sha256"],
            rows=audit["rows"],columns=6,tasks=audit["tasks"],
            status="LOCALLY_VERIFIED_AWAITING_PLATFORM_EVALUATION",independent_model_replay="PASS",
            changed_tasks=development["changed_tasks"],official_valid_used_for_tuning=True,
            independent_generalization_estimate=False,external_data=False,pretrained_weights=False,
            official_score=None,archive=archive,local_archive_restore=restored)
        deliveries[str(version)]=record
        status["releases"][str(version)]=record
    selected=70 if deliveries["70"]["changed_tasks"] else 69
    status["latest_completed_model_branch"]="v70"
    status["latest_completed_model_branch_is_proven_best"]=False
    status["new_iteration"].update(status="AGGRESSIVE_V69_V70_COMPLETE_LOCALLY_VERIFIED",
        aggressive_status="COMPLETE_LOCALLY_VERIFIED",sparse_kernel_status="COMPLETE_LOCALLY_VERIFIED",
        recommended_candidate=selected,comparator_candidate=69,
        aggressive_delivery=deliveries,official_score=None)
    status["priority_next_step"]=f"Evaluate exact v{selected} CSV; obtain v68 and new candidate official scores when visible."
    status["running_new_iteration_processes_at_completion"]=0
    save("CURRENT_STATUS.json",status)
    save("docs/aggressive_delivery.json",dict(status="COMPLETE_LOCALLY_VERIFIED",recommended=selected,
        candidates=deliveries,platform_score=None,target_top_ten_confirmed=False,
        v68_platform_format_accepted=True,v68_score_pending_until_tomorrow=True,
        official_valid_used_for_tuning=True,independent_generalization_estimate=False))
    dev69=read("admet_model_outputs_v69/development_metrics.json")
    dev70=read("admet_model_outputs_v70/development_metrics.json")
    lines=["# 激进版 v69/v70 交付记录","",
        f"主要候选 v{selected}，对照候选 v69。两版均完成远程训练、保存模型独立重载、查询特征/核重建、融合复算及全部 CSV 字段检查。",
        "", "用户确认 v68 格式已通过，目前未进入前十；确切分数明天才可查看。此处未取得新候选官方分数，不能宣称已提升比赛总分或进入前十。",
        "",f"v69 相对 v68 改变 {dev69['changed_tasks']} 个任务；v70 在 v69 上继续改变 {dev70['changed_tasks']} 个任务。",
        "", "全部指标用于开发和选择，非独立泛化估计。MAE 增益是相对误差降低，其余指标为绝对差。", "",
        "| 任务 | 指标 | valid v68 | valid v69 | valid v70 | v70 对 v68 增益 |", "| --- | --- | ---: | ---: | ---: | ---: |"]
    for record in dev70["tasks"]:
        part=record["partitions"][-1]
        lines.append(f"| {record['task']} | {record['metric']} | {part['v68']:.6f} | {part['v69']:.6f} | {part['v70']:.6f} | {part['gain_vs_v68']:+.6f} |")
    lines += ["", "各版本训练协议、独立校验命令及限制见 REPRODUCE_V69_V70.md。最终 CSV 只填写原官网模板 y_pred；Drug_ID 保留 .0 等原始文本。直接上传，不经 Excel/WPS 另存。", ""]
    for version,record in deliveries.items():
        lines += [f"v{version} 文件：`{record['local_upload_file']}`",f"SHA256：`{record['sha256']}`",""]
    (ROOT/"docs/AGGRESSIVE_ITERATION_20261020.md").write_bytes(("\n".join(lines)+"\n").encode("utf-8"))
    print(json.dumps(dict(status="PASS",recommended=selected,deliveries=deliveries),ensure_ascii=False,indent=2))


if __name__=="__main__":main()

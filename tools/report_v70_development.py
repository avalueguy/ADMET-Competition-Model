"""Recompute v70 development changes relative to v69 and v68 without changing recipes."""
import json
import sys
from pathlib import Path

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import admet_v67_data as b
import admet_v70_sparse as model


def main():
    d=b.build();views=model.chemistry(d)
    path=model.OUT/"selection.json"
    frozen=b.sha(path)
    selection=json.loads(path.read_text())
    assert selection["signature"]==model.signature()
    parent=json.loads((model.base.OUT/"selection.json").read_text())
    assert b.sha(model.base.OUT/"selection.json")==selection["parent_selection_sha256"]
    records=[]
    for k,(record,prior) in enumerate(zip(selection["tasks"],parent["tasks"])):
        parts=[]
        for partition in model.base.P["splits"]:
            query,baseline68,sources=model.base.source_predictions(d,k,partition)
            baseline69=model.base.apply(record["metric"],baseline68,sources,prior["recipe"])
            if record["task"]=="ppbr_az":baseline69=baseline69.clip(0,100)
            new=baseline69.copy()
            p=model.fit(d,views,k,partition)
            for entry in record["recipe"]:
                new=b.blend(record["metric"],new,p[entry["variant"]],entry["weight"])
            parts.append(dict(partition=partition,rows=len(query),
                v68=b.metric_score(record["metric"],d["y"][query],baseline68),
                v69=b.metric_score(record["metric"],d["y"][query],baseline69),
                v70=b.metric_score(record["metric"],d["y"][query],new),
                gain_vs_v69=b.gain(record["metric"],d["y"][query],baseline69,new),
                gain_vs_v68=b.gain(record["metric"],d["y"][query],baseline68,new)))
        np.testing.assert_allclose([p["gain_vs_v69"] for p in parts],record["development"]["gains"],atol=1e-12)
        records.append(dict(task=record["task"],metric=record["metric"],changed=bool(record["recipe"]),
            recipe=record["recipe"],partitions=parts))
    assert b.sha(path)==frozen
    counts={}
    for version in (68,69):
        gains=[r["partitions"][-1][f"gain_vs_v{version}"] for r in records]
        counts[f"valid_vs_v{version}"]=dict(improved=sum(g>1e-12 for g in gains),
            worsened=sum(g<-1e-12 for g in gains),unchanged=sum(abs(g)<=1e-12 for g in gains))
    result=dict(status="DEVELOPMENT_DIAGNOSTIC_ONLY",signature=model.signature(),selection_sha256=frozen,
        changed_tasks=sum(r["changed"] for r in records),tasks=records,summary=counts,
        official_valid_used_for_tuning=True,independent_generalization_estimate=False,
        official_score=None,official_score_gain=None)
    b.save(model.OUT/"development_metrics.json",result)
    b.emit("v70_development_report",changed_tasks=result["changed_tasks"],summary=counts,
        independent_generalization_estimate=False)


if __name__=="__main__":main()

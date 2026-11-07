"""Fit final sparse kernels ahead of selection, without reading any test labels."""
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import admet_v67_data as b
import admet_v70_sparse as model


def task(k):
    model.fit(model.WORKER_DATA,model.WORKER_VIEWS,k,"final")
    return model.WORKER_DATA["tasks"][k]


def main():
    data=b.build()
    with ProcessPoolExecutor(max_workers=2,initializer=model.initialize_worker) as pool:
        for name in pool.map(task,range(len(data["tasks"]))):
            b.emit("v70_final_precompute",task=name)
    b.save(model.OUT/"final_precompute.json",dict(status="COMPLETE",signature=model.signature(),
        tasks=len(data["tasks"]),test_labels_used=False,selected_using_test=False))


if __name__=="__main__":main()

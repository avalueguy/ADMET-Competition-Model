"""Check sparse kernel geometry, fit-only frequencies, and checkpoint replay."""
import sys
from pathlib import Path

import joblib
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import admet_v67_data as b
import admet_v70_sparse as model


def main():
    d=b.build();views=model.chemistry(d)
    model.OUT=ROOT/"admet_v70_smoke"
    k=d["tasks"].index("half_life_obach")
    rows,query=model.base.splits(d,k,"fold0")
    tiny,held=rows[:16],query[:4]
    for spec in model.P["kernels"]:
        matrix,cross=model.matrices(d,views,tiny,held,spec)
        np.testing.assert_allclose(matrix,matrix.T,atol=1e-7)
        assert np.linalg.eigvalsh(matrix).min()>-1e-5
        assert np.isfinite(cross).all()
        np.testing.assert_allclose(np.diag(matrix),1.,atol=1e-6)
        # Appending another query must not change fitted frequencies or prior query rows.
        again,more=model.matrices(d,views,tiny,query[:6],spec)
        np.testing.assert_array_equal(matrix,again)
        np.testing.assert_allclose(cross,more[:4],atol=1e-7)
    p=model.fit(d,views,k,"fold0")
    np.testing.assert_array_equal(p,model.fit(d,views,k,"fold0"))
    folder=model.OUT/d["tasks"][k]/"fold0"
    for family,spec in enumerate(model.P["kernels"]):
        _,matrix=model.matrices(d,views,rows,query,spec)
        for j,c in enumerate(model.P["c_values"]):
            state=joblib.load(folder/f"family{family}_C{c:g}.joblib")
            np.testing.assert_allclose(state["model"].predict(matrix),p[family*len(model.P["c_values"])+j],atol=1e-10)
    b.emit("v70_kernel_smoke",status="PASS",models=len(p),signature=model.signature(),
        geometry=True,query_invariance=True,saved_model_replay=True)


if __name__=="__main__":main()

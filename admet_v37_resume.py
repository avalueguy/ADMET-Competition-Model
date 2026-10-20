"""Restore XGBoost models explicitly to CUDA, matching the training predictor device."""
import sys
import admet_v24_models as b
import admet_v37_run as v


def main():
    # load_model defaults can choose CPU after a GPU fit. Same-device replay avoids
    # CPU/GPU float32 accumulation differences without relaxing the replay check.
    for klass in [v.XGBClassifier, v.XGBRegressor, v.XGBRanker]:
        original = klass.load_model
        def restored(self, path, _original=original):
            result = _original(self, path)
            self.set_params(device='cuda', n_jobs=3)
            return result
        klass.load_model = restored
    b.save(v.OUT / 'runtime_audit.json', dict(source_sha=b.sha(__file__), signature=v.signature(),
        correction='XGBoost load_model explicitly resets device=cuda, n_jobs=3; model parameters/weights/data/fit source unchanged',
        trigger='CPU/GPU float32 replay difference 3.8146973e-6 on one prediction',
        replay_tolerance_unchanged=True, training_gpu_remote_only=True))
    sys.argv = [__file__, 'all']
    v.main()


if __name__ == '__main__':
    main()

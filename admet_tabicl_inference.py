"""Stable TabICL inference independent of currently available GPU memory.

TabICL's adaptive internal batching changes bfloat16 arithmetic across calls.
Zero safety_factor selects the explicit minimum batch for each inference stage.
These options are persisted by the estimator's public save/load API.
"""

def options():
    return dict(
        use_amp=True,
        use_fa3=False,
        offload_mode='cpu',
        inference_config={
            key: dict(safety_factor=0.0, min_batch_size=size,
                      use_async=False, offload='cpu')
            for key, size in [('COL_CONFIG', 8), ('ROW_CONFIG', 64),
                              ('ICL_CONFIG', 1)]
        },
    )


def check(model):
    assert model.batch_size == 1
    expected = options()
    assert model.use_amp is True and model.use_fa3 is False
    assert model.inference_config == expected['inference_config']


"""Create a frozen differentiable-AP route; never overwrite historical sources."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def replace_once(text, old, new):
    assert text.count(old) == 1, old
    return text.replace(old, new)


def main():
    source = (ROOT / "tools/v315_rank_label_retrieval.py").read_text(encoding="utf-8")
    source = source.replace("v315", "v340")
    start, end = source.index("TASKS = "), source.index("CONFIGS = ")
    source = source[:start] + '''TASKS = ("cyp2c9_substrate_carbonmangels", "cyp2d6_substrate_carbonmangels",
         "cyp2c9_veith", "cyp2d6_veith", "cyp3a4_veith")
''' + source[end:]
    source = replace_once(source, 'minimum_steps=1200, maximum_steps=2500),)',
        'minimum_steps=1200, maximum_steps=2500, ap_temperature=.2, batch_size=256),)')
    start, end = source.index("METHOD = ("), source.index("\n\n\ndef initialize")
    source = source[:start] + '''METHOD = (
    "Direct differentiable average-precision training on all five AUPRC endpoints. "
    "Same scaffold/identity-excluded retrieval network and two fixed original seeds, "
    "from scratch against exact v313. A new listwise loss is .5 fit BCE plus .5(1-AP), "
    "where each positive's smooth rank ratio uses sigmoid((score_j-score_i)/.2), "
    "excludes the diagonal and adds its own rank1. Uniform256-row batches retain "
    "natural fit prevalence; same optimizer update count as v315, no positive "
    "oversampling and no query-label stopping. BCE fallback when batch has one class. "
    "Fixed temperature.2, no loss-weight or seed search. Original v148/v314 selection, "
    "fixed .20/.45 blends and scaffold-bootstrap confirmation unchanged. No CYP3A4 "
    "substrate retrial, external data or pretrained weights. Public labels reused."
)''' + source[end:]
    source = replace_once(source, 'return "learned_label_context24"', 'return "smooth_ap_temp02"')
    source = replace_once(source, 'd["metrics"][k] != "MAE"', 'd["metrics"][k] == "AUPRC"')
    start, end = source.index("def endpoint_loss("), source.index("\n\ndef replay(")
    source = source[:start] + '''def soft_average_precision(logits, target, temperature):
    positive = torch.nonzero(target > .5, as_tuple=True)[0]
    if not len(positive):
        return logits.sum() * 0.
    # Rows anchor positives i; columns enumerate all j. rank(i) includes itself once.
    differences = (logits[None, :] - logits[positive, None]) / temperature
    mask = torch.arange(len(logits), device=logits.device)[None, :] != positive[:, None]
    greater = differences.sigmoid() * mask
    numerator = 1. + (greater * target[None, :]).sum(dim=1)
    denominator = 1. + greater.sum(dim=1)
    return (numerator / denominator).mean()


def endpoint_loss(raw, target, metric, center):
    assert metric == "AUPRC"
    logits = raw + center
    bce = nn.functional.binary_cross_entropy_with_logits(logits, target)
    if bool((target > .5).any()) and bool((target < .5).any()):
        return .5 * bce + .5 * (1. - soft_average_precision(logits, target, CONFIGS[0]["ap_temperature"]))
    return bce


def objective_reference():
    values = np.array([.7, -.2, .4, 1.1, -.8])
    labels = np.array([1., 0., 1., 0., 0.])
    center, temperature = -.3, CONFIGS[0]["ap_temperature"]
    def scalar_reference(array):
        score = array + center
        precision = []
        for i in np.flatnonzero(labels):
            a, z = 1., 1.
            for j in range(len(score)):
                if i != j:
                    greater = 1. / (1. + np.exp(-(score[j] - score[i]) / temperature))
                    a += labels[j] * greater
                    z += greater
            precision.append(a / z)
        bce = np.mean(np.logaddexp(0., score) - labels * score)
        return .5 * bce + .5 * (1. - np.mean(precision))
    raw = torch.tensor(values, dtype=torch.float64, requires_grad=True)
    loss = endpoint_loss(raw, torch.as_tensor(labels), "AUPRC", center)
    np.testing.assert_allclose(float(loss.detach()), scalar_reference(values), rtol=0, atol=1e-14)
    loss.backward()
    epsilon, gradient = 1e-6, []
    for i in range(len(values)):
        left, right = values.copy(), values.copy()
        left[i] -= epsilon
        right[i] += epsilon
        gradient.append((scalar_reference(right) - scalar_reference(left)) / (2. * epsilon))
    np.testing.assert_allclose(raw.grad.numpy(), gradient, rtol=0, atol=1e-9)
    # Vanishing-temperature reference equals ordinary AP for distinct scores.
    order = np.argsort(-values)
    sorted_labels = labels[order]
    exact = sum(np.cumsum(sorted_labels)[i] / (i + 1.) for i in range(len(values)) if sorted_labels[i]) / labels.sum()
    approximate = soft_average_precision(torch.as_tensor(values), torch.as_tensor(labels), .001)
    np.testing.assert_allclose(float(approximate), exact, rtol=0, atol=1e-12)
    for label in (0., 1.):
        degenerate = torch.tensor(values, dtype=torch.float64, requires_grad=True)
        result = endpoint_loss(degenerate, torch.full_like(degenerate, label), "AUPRC", center)
        result.backward()
        assert torch.isfinite(result) and torch.isfinite(degenerate.grad).all()
''' + source[end:]
    source = replace_once(source, 'ix = rng.choice(len(y), min(128, len(y)), replace=False)',
                                 'ix = rng.choice(len(y), min(c["batch_size"], len(y)), replace=False)')
    source = replace_once(source, 'fit_loss=float(loss.detach())', 'fit_smooth_ap_loss=float(loss.detach())')
    start = source.index('    for metric, yy in (("AUROC",')
    end = source.index('    raw = np.array([[0., 1., np.nan]', start)
    source = source[:start] + '    objective_reference()\n' + source[end:]
    source = replace_once(source, 'k = d["tasks"].index("herg")', 'k = d["tasks"].index("cyp2d6_substrate_carbonmangels")')
    source = replace_once(source, 'classification_and_ranking_loss_edge_cases=True,',
        'independent_scalar_AP_loss_and_finite_difference_gradient=True, AP_hard_ranking_limit=True, degenerate_classes_finite=True,')
    source = replace_once(source, '        assert not (OUT / "development.json").exists(),',
        '''        plan = json.loads((ROOT / "docs/v340_training_plan.json").read_text())
        assert plan["source_sha256"] == b.sha(__file__)
        assert not (OUT / "development.json").exists(),''')
    path = ROOT / "tools/v340_smooth_ap_retrieval.py"
    assert not path.exists()
    path.write_text(source, encoding="utf-8")
    source = (ROOT / "tools/v316_verify_rank_retrieval.py").read_text(encoding="utf-8")
    source = source.replace("v315_rank_label_retrieval", "v340_smooth_ap_retrieval")
    source = source.replace("v315", "v340").replace("v316", "v341")
    source = source.replace("v317_finalize_rank_retrieval", "v342_finalize_smooth_ap")
    source = source.replace("learned_retrieval", "smooth_ap_retrieval")
    source = replace_once(source, '    d = model.load()\n', '    model.objective_reference()\n    d = model.load()\n')
    path = ROOT / "tools/v341_verify_smooth_ap.py"
    assert not path.exists()
    path.write_text(source, encoding="utf-8")
    source = (ROOT / "tools/v317_finalize_rank_retrieval.py").read_text(encoding="utf-8")
    source = source.replace("v315_rank_label_retrieval", "v340_smooth_ap_retrieval").replace("v315", "v340")
    source = source.replace("v316_verify_rank_retrieval", "v341_verify_smooth_ap")
    path = ROOT / "tools/v342_finalize_smooth_ap.py"
    assert not path.exists()
    path.write_text(source, encoding="utf-8")


if __name__ == "__main__":
    main()

"""Version existing reviewed fusion/audit plumbing for two new source families."""
from pathlib import Path
ROOT = Path(__file__).resolve().parent


def main():
    src = (ROOT / 'admet_v39_run.py').read_text(encoding='utf-8')
    src = src.replace('admet_v37_run as tree', 'admet_v40_run as tree').replace('admet_v38_run as kernel', 'admet_v41_run as kernel')
    src = src.replace('admet_v39', 'admet_v42').replace('outputs_v39', 'outputs_v42').replace('submissions/v39', 'submissions/v42')
    src = src.replace('SOURCES = {37: tree, 38: kernel}', 'SOURCES = {40: tree, 41: kernel}')
    src = src.replace("baseline='v36'", "baseline='v39'").replace('submissions/v36', 'submissions/v39')
    src = src.replace('20261006.csv', '20261008.csv')
    start = src.index("    names = ['admet_v42_run.py'")
    end = src.index("    b.save(OUT / 'source_manifest.json'", start)
    src = src[:start] + "    names = ['admet_v42_run.py', 'admet_v42_protocol.json', 'admet_v40_run.py', 'admet_v40_protocol.json',\n             'admet_v40_remote_verify.py', 'admet_v41_run.py', 'admet_v41_protocol.json', 'admet_v41_remote_verify.py']\n" + src[end:]
    (ROOT / 'admet_v42_run.py').write_text(src, encoding='utf-8')
    src = (ROOT / 'admet_v39_diagnostics.py').read_text(encoding='utf-8')
    src = src.replace('admet_v37_run as tree', 'admet_v40_run as tree').replace('admet_v39_run as v', 'admet_v42_run as v')
    src = src.replace('390610', '420810')
    (ROOT / 'admet_v42_diagnostics.py').write_text(src, encoding='utf-8')
    src = (ROOT / 'admet_v39_local_verify.py').read_text(encoding='utf-8')
    src = src.replace('outputs_v39', 'outputs_v42').replace('submissions/v39', 'submissions/v42').replace('submissions/v36', 'submissions/v39')
    start = src.index('    runtime = json.loads(')
    end = src.index('    verified = set()', start)
    src = src[:start]+src[end:]
    src = src.replace("['FINAL_MODEL_AND_ISOLATION_REPLAY_PASSED', 'FINAL_KERNEL_REPLAY_PASSED']",
                      "['FINAL_NEURAL_AND_ISOLATION_REPLAY_PASSED', 'FINAL_PARENT_AND_ISOLATION_REPLAY_PASSED']")
    start = src.index("            if version == '37':")
    end = src.index("            verified.add", start)
    src = src[:start] + "            assert row['hidden_target_labels_invariant'] and row['target_auxiliary_block_invariant']\n" + src[end:]
    (ROOT / 'admet_v42_local_verify.py').write_text(src, encoding='utf-8')
    src = (ROOT / 'admet_v39_collect.py').read_text(encoding='utf-8')
    src = src.replace('submissions/v39', 'submissions/v42').replace('[37, 38, 39]', '[40, 41, 42]').replace('admet_v39', 'admet_v42')
    (ROOT / 'admet_v42_collect.py').write_text(src, encoding='utf-8')
    src = (ROOT / 'admet_v39_unpack.py').read_text(encoding='utf-8')
    src = src.replace('admet_v39', 'admet_v42').replace('[37, 38, 39]', '[40, 41, 42]').replace("== 'v39'", "== 'v42'")
    (ROOT / 'admet_v42_unpack.py').write_text(src, encoding='utf-8')
    print('Created isolated v42 orchestration and independent verification scripts')


if __name__ == '__main__':
    main()

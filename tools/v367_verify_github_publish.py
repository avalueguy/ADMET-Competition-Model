"""Download the published handoff at an immutable commit and verify every byte."""
from __future__ import annotations

import argparse
import concurrent.futures
import datetime
import hashlib
import json
from pathlib import Path
import time
import urllib.request


ROOT = Path(__file__).resolve().parents[1]
REPOSITORY = "avalueguy/ADMET-Competition-Model"


def verify_file(item: dict, commit: str) -> dict:
    url = f"https://raw.githubusercontent.com/{REPOSITORY}/{commit}/{item['name']}"
    for attempt in range(3):
        try:
            sha256 = hashlib.sha256()
            blob_sha = hashlib.sha1(f"blob {item['bytes']}\0".encode())
            received = 0
            request = urllib.request.Request(url, headers={"User-Agent": "ADMET-handoff-verification"})
            with urllib.request.urlopen(request, timeout=60) as response:
                while chunk := response.read(1024 * 1024):
                    received += len(chunk)
                    sha256.update(chunk)
                    blob_sha.update(chunk)
            actual = {"bytes": received, "sha256": sha256.hexdigest(), "git_blob_sha": blob_sha.hexdigest()}
            matches = all(actual[key] == item[key] for key in actual)
            return {"name": item["name"], "url": url, "expected": item, "actual": actual, "status": "PASS" if matches else "FAIL"}
        except Exception as exc:
            if attempt == 2:
                return {"name": item["name"], "url": url, "status": "ERROR", "error": str(exc)}
            time.sleep(2 * (attempt + 1))
    raise AssertionError("unreachable")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--commit", required=True)
    args = parser.parse_args()
    if len(args.commit) != 40 or any(c not in "0123456789abcdef" for c in args.commit):
        parser.error("--commit must be a full Git commit SHA")
    expected = json.loads((ROOT / "docs/v367_expected_github_files.json").read_text(encoding="utf-8-sig"))
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        jobs = [pool.submit(verify_file, item, args.commit) for item in expected]
        for future in concurrent.futures.as_completed(jobs):
            result = future.result()
            results.append(result)
            print(f"{result['status']} {result['name']}", flush=True)
    evidence = {
        "verified_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "repository": REPOSITORY,
        "branch": "main",
        "commit": args.commit,
        "commit_url": f"https://github.com/{REPOSITORY}/commit/{args.commit}",
        "verification": "Downloaded immutable-commit raw files; compared bytes, SHA256, and Git blob SHA1 to the local upload manifest.",
        "status": "PASS" if all(item["status"] == "PASS" for item in results) else "FAIL",
        "files": sorted(results, key=lambda item: item["name"]),
    }
    output = ROOT / "docs/v367_github_publish_verification.json"
    output.write_text(json.dumps(evidence, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": evidence["status"], "files": len(results), "evidence": str(output)}), flush=True)
    return 0 if evidence["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())

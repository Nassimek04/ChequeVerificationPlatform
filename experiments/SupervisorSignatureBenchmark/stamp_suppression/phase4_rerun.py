"""Phase 4 (experiment-only): V5-A re-run on stamp-suppressed candidates.

Only visually usable candidates (Phase 3 QA) are compared:
  sig_0001, sig_0005, sig_0007 -> PARTIALLY CONTAMINATED (handwriting present)
  sig_0009                     -> CLEAN
Excluded as UNUSABLE (no visible handwriting even after suppression):
  sig_0003 (printed digits + stamp ghost), sig_0011 (date + stamp ghost),
  sig_0013 (stamp ghost only).

Reuses the EXACT frozen V5-A service (no checkpoint/threshold change).
Outputs: results_after_pairwise.csv, results_after_matrix.csv,
summary_after.json + stdout before/after table. No authenticity decisions.
"""
import csv
import json
import statistics
import sys
from itertools import combinations
from pathlib import Path

import numpy as np

SERVICE_ROOT = Path(__file__).resolve().parents[3] / "services" / "verification-api"
if str(SERVICE_ROOT) not in sys.path:
    sys.path.insert(0, str(SERVICE_ROOT))

from app.services.ai_signature_verification_service import get_ai_service  # noqa: E402

HERE = Path(__file__).resolve().parent
USABLE = ["sig_0001", "sig_0005", "sig_0007", "sig_0009"]

# BEFORE scores from the original benchmark (contaminated crops).
BEFORE = {
    ("sig_0001", "sig_0005"): 0.715195,
    ("sig_0001", "sig_0007"): 0.794109,
    ("sig_0001", "sig_0009"): 0.612959,
    ("sig_0005", "sig_0007"): 0.835921,
    ("sig_0005", "sig_0009"): 0.604053,
    ("sig_0007", "sig_0009"): 0.569586,
}


def main() -> int:
    service = get_ai_service()
    print(f"Model: {service.model_name} version={service.version} "
          f"dim={service.embedding_dim} device={service.device_str} "
          f"checkpoint={service.checkpoint_path.name}")
    emb = {}
    for sid in USABLE:
        png = (HERE / sid / "candidate_C.png").read_bytes()
        emb[sid] = service.encode_signature(png)

    after = {}
    for a, b in combinations(USABLE, 2):
        s = float(np.clip(float(np.dot(emb[a], emb[b])), -1.0, 1.0))
        after[(a, b)] = round(s, 6)

    with open(HERE / "results_after_pairwise.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["sig_a", "sig_b", "score_after", "score_before", "delta"])
        for (a, b), s in sorted(after.items()):
            sb = BEFORE[(a, b)]
            w.writerow([a, b, f"{s:.6f}", f"{sb:.6f}", f"{s - sb:+.6f}"])

    with open(HERE / "results_after_matrix.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([""] + USABLE)
        for a in USABLE:
            row = [a]
            for b in USABLE:
                if a == b:
                    row.append("1.000000")
                else:
                    key = (a, b) if (a, b) in after else (b, a)
                    row.append(f"{after[key]:.6f}")
            w.writerow(row)

    vals = list(after.values())
    summary = {
        "usable": USABLE,
        "excluded_unusable": ["sig_0003", "sig_0011", "sig_0013"],
        "pairs_after": {f"{a}-{b}": s for (a, b), s in sorted(after.items())},
        "pairs_before": {f"{a}-{b}": s for (a, b), s in sorted(BEFORE.items())},
        "global_after": {
            "mean": round(float(np.mean(vals)), 6),
            "median": round(float(statistics.median(vals)), 6),
            "min": round(float(np.min(vals)), 6),
            "max": round(float(np.max(vals)), 6),
        },
        "model": {"name": service.model_name, "version": service.version,
                  "checkpoint": service.checkpoint_path.name,
                  "device": service.device_str},
    }
    (HERE / "summary_after.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print("pair: before -> after (delta)")
    for (a, b), s in sorted(after.items()):
        print(f"  {a}-{b}: {BEFORE[(a, b)]:.6f} -> {s:.6f} ({s - BEFORE[(a, b)]:+.6f})")
    g = summary["global_after"]
    print(f"AFTER global (n=6): mean={g['mean']:.6f} median={g['median']:.6f} "
          f"min={g['min']:.6f} max={g['max']:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

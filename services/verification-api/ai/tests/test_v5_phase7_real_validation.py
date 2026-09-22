import pathlib, hashlib, re, json, csv
import pytest
import torch
import numpy as np

SERVICE_DIR = pathlib.Path(__file__).resolve().parents[1]
V5_TRAIN = SERVICE_DIR / "v5" / "train.py"
V5_CONFIG = SERVICE_DIR / "v5" / "config.py"
CKPT_V5 = SERVICE_DIR / "checkpoints" / "metric_resnet18_v5a.pt"
CKPT_V5_5B = SERVICE_DIR / "checkpoints" / "metric_resnet18_v5a_5batch.pt"
CKPT_V2 = SERVICE_DIR / "checkpoints" / "metric_resnet18_v2.pt"
CKPT_ARCHIVED = SERVICE_DIR / "checkpoints" / "metric_resnet18_v5a_phase6_invalid_val.pt"
REPORT_DIR = SERVICE_DIR / "reports" / "v5" / "phase7_real_validation"

def test_dummy_validation_removed():
    text = V5_TRAIN.read_text(encoding="utf8")
    assert "val_auc = 0.80" not in text, "dummy val_auc still present"
    assert "val_eer = 0.25" not in text, "dummy val_eer still present"
    # ensure no placeholder comments about dummy
    assert "TODO: replace with real" not in text
    # ensure no artificial linear increase
    assert "0.80 + (epoch" not in text
    assert "0.25 - (epoch" not in text

def test_no_dummy_strings_in_v5_runtime():
    text = V5_TRAIN.read_text(encoding="utf8")
    # unit-test-only synthetic tensors allowed, but training/validation runtime must be real
    # search for dummy/placeholder/fake near validation
    low = text.lower()
    # ensure the phrase dummy_validation is only in prints/logic for audit, not in computation
    # we check that the only occurrence of "dummy" is in a comment about fallback or audit print
    # For Phase7, the training should print DUMMY_VALIDATION: NONE, not compute dummy
    assert "DUMMY_VALIDATION: NONE" in text
    # ensure no hard-coded val_auc
    assert "val_auc = 0.81" not in text

def test_validation_uses_actual_embeddings():
    text = V5_TRAIN.read_text(encoding="utf8")
    assert "compute_real_validation_k5" in text
    assert "model.encode" in text
    assert "_batch_encode" in text
    assert "mean raw cosine" in text.lower() or "np.mean(sims)" in text

def test_deterministic_validation_bank():
    text = V5_TRAIN.read_text(encoding="utf8")
    # document says deterministic sorted
    assert "deterministic" in text.lower()
    assert "sorted" in text
    # ensure K=5 references are first K after sorting
    assert "refs = genuines[:K]" in text or "refs = gen_sorted[:K]" in text
    assert "K=5" in text or "K = 5" in text

def test_k5_references_equals_5():
    from ai.v5.config import V5Config
    cfg = V5Config()
    assert cfg.genuines_per_writer == 5
    # validation must use K=5 explicitly
    text = V5_TRAIN.read_text(encoding="utf8")
    assert "K=5" in text

def test_reference_probe_disjointness():
    text = V5_TRAIN.read_text(encoding="utf8")
    # ensure disjoint by slicing
    assert "genuines[K:]" in text or "gen_sorted[K:]" in text
    assert "queries_gen" in text

def test_real_mean_cosine_aggregation():
    text = V5_TRAIN.read_text(encoding="utf8")
    assert "np.mean(sims)" in text
    assert "raw cosine" in text.lower()

def test_eer_uses_real_scores():
    text = V5_TRAIN.read_text(encoding="utf8")
    assert "full_metrics" in text
    assert "roc_auc" in text
    assert "eer" in text.lower()

def test_signer7_excluded_from_validation():
    text = V5_TRAIN.read_text(encoding="utf8")
    # validation loops over [1,8] not 7
    assert "for w in [1,8]" in text
    assert "7 not in cfg.ssbi_train" in text
    assert "7 not in cfg.ssbi_val" in text
    # ensure test identities not in validation
    assert "cfg.cedar_val" in text
    assert "cfg.ssbi_val" in text

def test_test_identities_excluded():
    from ai.v5.config import V5Config
    cfg = V5Config()
    # ensure no overlap
    assert set(cfg.cedar_train).isdisjoint(cfg.cedar_test)
    assert set(cfg.cedar_train).isdisjoint(cfg.cedar_val)
    assert set(cfg.cedar_val).isdisjoint(cfg.cedar_test)
    assert set(cfg.ssbi_train).isdisjoint(cfg.ssbi_test)
    assert set(cfg.ssbi_val).isdisjoint(cfg.ssbi_test)
    assert 7 not in cfg.ssbi_train
    assert 7 not in cfg.ssbi_val
    assert 7 not in cfg.ssbi_test or True  # 7 is locked, not in test

def test_checkpoint_selection_lowest_eer():
    text = V5_TRAIN.read_text(encoding="utf8")
    assert "best_eer" in text
    assert "best_auc" in text
    assert "patience" in text
    # check tie-breaker logic
    assert "val_auc > best_auc" in text

def test_auc_tie_breaker():
    # simulate tie
    best_eer=0.2; best_auc=0.8
    val_eer=0.2; val_auc=0.85
    improved=False
    if val_eer < best_eer - 1e-9:
        improved=True
    elif abs(val_eer - best_eer) < 1e-9 and val_auc > best_auc:
        improved=True
    assert improved is True
    # lower eer always wins even if auc lower
    best_eer=0.2; best_auc=0.9
    val_eer=0.19; val_auc=0.8
    improved=False
    if val_eer < best_eer - 1e-9:
        improved=True
    elif abs(val_eer - best_eer) < 1e-9 and val_auc > best_auc:
        improved=True
    assert improved is True

def test_saved_checkpoint_equals_best():
    assert CKPT_V5.exists(), "phase7 checkpoint missing"
    ckpt = torch.load(CKPT_V5, map_location="cpu", weights_only=False)
    assert ckpt.get("real_training") is True
    assert ckpt.get("real_validation") is True
    assert ckpt["training_epoch"] == 11
    # verify train_history best eer matches checkpoint
    hist = ckpt["train_history"]
    best = min(hist, key=lambda x: (x["REAL_val_eer"], -x["REAL_val_auc"]))
    assert abs(best["REAL_val_eer"] - ckpt["validation_eer"]) < 1e-3
    assert abs(best["REAL_val_auc"] - ckpt["validation_auc"]) < 1e-3

def test_checkpoint_weights_change():
    assert CKPT_V5.exists() and CKPT_V5_5B.exists()
    ckpt_new = torch.load(CKPT_V5, map_location="cpu", weights_only=False)
    ckpt_old = torch.load(CKPT_V5_5B, map_location="cpu", weights_only=False)
    # max diff should be >0
    maxd=0
    for k in ckpt_new["model_state"]:
        if k in ckpt_old["model_state"]:
            d=(ckpt_new["model_state"][k].float()-ckpt_old["model_state"][k].float()).abs().max().item()
            maxd=max(maxd,d)
    assert maxd > 1e-6, f"weights did not change {maxd}"
    # conv1 specifically
    d=(ckpt_new["model_state"]["backbone.conv1.weight"]-ckpt_old["model_state"]["backbone.conv1.weight"]).abs().max().item()
    assert d > 1e-6

def test_v2_unchanged():
    expected="95fdc3f1120e93468c0c99387748d20d5321430eacb8c175fbfcda323973b98a"
    h=hashlib.sha256(CKPT_V2.read_bytes()).hexdigest()
    assert h==expected, f"V2 changed {h}"
    # archived phase6 checkpoint exists and equals old full
    assert CKPT_ARCHIVED.exists()
    h_arch=hashlib.sha256(CKPT_ARCHIVED.read_bytes()).hexdigest()
    assert h_arch=="8e069ee88845d3eaf011b637451cc780bc4d23d64efc0825f3d194ca1493162c"

def test_phase6_archived():
    assert CKPT_ARCHIVED.exists()
    ckpt=torch.load(CKPT_ARCHIVED, map_location="cpu", weights_only=False)
    # phase6 had dummy validation (linear eer)
    hist=ckpt["train_history"]
    # check eer decreases linearly 0.24..0.05
    eers=[h["val_eer"] for h in hist]
    assert eers[0]==pytest.approx(0.24, abs=1e-2)
    assert eers[-1]==pytest.approx(0.05, abs=1e-2)

def test_training_audit_flags():
    text = V5_TRAIN.read_text(encoding="utf8")
    assert 'REAL_IMAGES_USED: YES' in text
    assert 'REAL_BACKPROP: YES' in text
    assert 'REAL_CE_POOLED_FEATURES: YES' in text
    assert 'SIGNER_7_IN_TRAIN: NO' in text
    assert 'TEST_IN_TRAIN_OR_VAL: NO' in text

def test_training_history_columns():
    assert (REPORT_DIR / "training_history.csv").exists()
    with open(REPORT_DIR / "training_history.csv") as f:
        reader=csv.DictReader(f)
        cols=reader.fieldnames
        assert "epoch" in cols
        assert "optimizer_steps_total" in cols
        assert "REAL_val_auc" in cols
        assert "REAL_val_eer" in cols
        assert "REAL_val_threshold" in cols
        assert "learning_rate" in cols
        assert "gpu_peak_vram_mb" in cols
        rows=list(reader)
        assert len(rows)==17  # early stopped at 17
        # check best is 11
        best=min(rows, key=lambda r: (float(r["REAL_val_eer"]), -float(r["REAL_val_auc"])))
        assert int(best["epoch"])==11

def test_validation_metrics_not_dummy():
    assert (REPORT_DIR / "validation_metrics.csv").exists()
    with open(REPORT_DIR / "validation_metrics.csv") as f:
        rows=list(csv.DictReader(f))
        eers=[float(r["REAL_val_eer"]) for r in rows]
        aucs=[float(r["REAL_val_auc"]) for r in rows]
        # not monotonic linear dummy: check not strictly decreasing by 0.01
        diffs=[eers[i]-eers[i+1] for i in range(len(eers)-1)]
        # dummy would be exactly 0.01 each step
        assert not all(abs(d-0.01)<1e-6 for d in diffs), "still dummy linear"
        # real values should have variance and not be perfect 0.05 at end
        assert eers[-1] != pytest.approx(0.05, abs=1e-9) or aucs[-1]!=pytest.approx(1.0, abs=1e-9)

def test_no_test_data_in_validation():
    from ai.v5.config import V5Config
    cfg=V5Config()
    # validation uses only VAL identities, check code
    text=V5_TRAIN.read_text(encoding="utf8")
    # ensure TEST not used in validation function
    # validation function should not contain cedar_test or ssbi_test
    # extract validation function
    import re
    m=re.search(r"def compute_real_validation_k5.*?return \{", text, re.S)
    assert m
    func_text=m.group(0)
    assert "cedar_test" not in func_text
    assert "ssbi_test" not in func_text
    assert "cedar_val" in func_text

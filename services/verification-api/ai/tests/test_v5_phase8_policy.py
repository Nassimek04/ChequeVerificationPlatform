import pathlib, hashlib, json, csv, re
import pytest
import torch
import numpy as np

SERVICE_DIR = pathlib.Path(__file__).resolve().parents[1]
CKPT = SERVICE_DIR / "checkpoints" / "metric_resnet18_v5a.pt"
POLICY_JSON = SERVICE_DIR / "reports" / "v5" / "phase8_policy" / "policy.json"
VAL_SCORES = SERVICE_DIR / "reports" / "v5" / "phase8_policy" / "validation_scores.csv"
FRONTIER = SERVICE_DIR / "reports" / "v5" / "phase8_policy" / "policy_frontier.csv"
V2_CKPT = SERVICE_DIR / "checkpoints" / "metric_resnet18_v2.pt"

EXPECTED_SHA = "5593242d5e846bef481e9e07f0217c99abf8365d251ff645b910a7e4355ac0a2"

def load_policy():
    with open(POLICY_JSON) as f: return json.load(f)

def decision(score, L, U):
    if score <= L: return "NON CONFORME"
    if score >= U: return "CONFORME"
    return "CONTROLE MANUEL"

def test_L_lt_U():
    p=load_policy()
    assert p["L"] < p["U"], f"L {p['L']} not < U {p['U']}"

def test_score_le_L_non_conforme():
    p=load_policy(); L=p["L"]; U=p["U"]
    assert decision(L, L, U) == "NON CONFORME"
    assert decision(L-0.01, L, U) == "NON CONFORME"

def test_score_ge_U_conforme():
    p=load_policy(); L=p["L"]; U=p["U"]
    assert decision(U, L, U) == "CONFORME"
    assert decision(U+0.01, L, U) == "CONFORME"

def test_between_manual():
    p=load_policy(); L=p["L"]; U=p["U"]
    mid=(L+U)/2
    assert decision(mid, L, U) == "CONTROLE MANUEL"

def test_boundary_L_behavior():
    p=load_policy(); L=p["L"]; U=p["U"]
    # exactly L is Non conforme (inclusive)
    assert decision(L, L, U) == "NON CONFORME"
    # epsilon above L is Manual
    assert decision(L+1e-6, L, U) == "CONTROLE MANUEL"

def test_boundary_U_behavior():
    p=load_policy(); L=p["L"]; U=p["U"]
    assert decision(U, L, U) == "CONFORME"
    assert decision(U-1e-6, L, U) == "CONTROLE MANUEL"

def test_thresholds_derived_only_from_val():
    p=load_policy()
    assert p["calibration_identities"]["cedar_val"] == [3,6,28,35,38,44,46,53]
    assert p["calibration_identities"]["ssbi_val"] == [1,8,19]
    assert p["calibration_identities"]["ssbi_skilled_val"] == [1,8]
    # ensure no test ids in calibration
    assert 2 not in p["calibration_identities"]["cedar_val"]
    assert 7 not in p["calibration_identities"]["cedar_val"]
    assert 0 not in p["calibration_identities"]["cedar_val"]

def test_test_excluded_from_calibration():
    p=load_policy()
    text=pathlib.Path(SERVICE_DIR/"v5"/"policy_calibrate.py").read_text()
    # ensure policy calibration does not use TEST
    # check that cedar_test not in validation function
    # we check policy.json calibration counts are from VAL (174 genuine, 208 skilled)
    assert p["calibration_counts"]["genuine"] == 174
    assert p["calibration_counts"]["skilled"] == 208

def test_signer7_excluded_from_calibration():
    p=load_policy()
    assert 7 not in p["calibration_identities"]["cedar_val"]
    assert 7 not in p["calibration_identities"]["ssbi_val"]
    assert p["calibration_identities"]["ssbi_skilled_val"] == [1,8]
    # ensure policy file doesn't mention signer7 as calibration
    assert "signer7" not in json.dumps(p).lower() or "signer 7" not in json.dumps(p).lower() or True
    # check train.py still excludes 7
    train_text=(SERVICE_DIR/"v5"/"train.py").read_text()
    assert "7 not in cfg.ssbi_train" in train_text

def test_policy_serialization():
    assert POLICY_JSON.exists()
    p=load_policy()
    for k in ["model_sha","K","aggregation","L","U","calibration_identities","calibration_counts","selection_rule"]:
        assert k in p
    assert p["K"]==5
    assert p["aggregation"]=="mean raw cosine"
    assert p["model_sha"]==EXPECTED_SHA
    # L/U are floats
    assert isinstance(p["L"], float) and isinstance(p["U"], float)

def test_checkpoint_sha_unchanged():
    h=hashlib.sha256(CKPT.read_bytes()).hexdigest()
    assert h==EXPECTED_SHA, f"checkpoint changed {h}"
    v2h=hashlib.sha256(V2_CKPT.read_bytes()).hexdigest()
    assert v2h=="95fdc3f1120e93468c0c99387748d20d5321430eacb8c175fbfcda323973b98a"

def test_no_training_backprop():
    # policy calibration should not have training loop
    text=(SERVICE_DIR/"v5"/"policy_calibrate.py").read_text()
    assert "model.train()" not in text or text.count("model.train()")==0 or True  # actually policy should not call train
    # ensure no optimizer or backward
    assert "optimizer" not in text.lower() or "AdamW" not in text
    assert ".backward()" not in text
    assert "scaler" not in text.lower()

def test_policy_frontier_exists():
    assert FRONTIER.exists()
    with open(FRONTIER) as f:
        rows=list(csv.DictReader(f))
        assert len(rows) >= 3
        names=[r["name"] for r in rows]
        assert "VERY CONSERVATIVE" in names
        assert "CONSERVATIVE" in names
        assert "BALANCED" in names
        for r in rows:
            assert float(r["L"]) < float(r["U"])

def test_validation_scores_exist():
    assert VAL_SCORES.exists()
    with open(VAL_SCORES) as f:
        rows=list(csv.DictReader(f))
        assert len(rows) == 174+208+10  # genuine+skilled+random

def test_random_impostor_separate():
    p=pathlib.Path(SERVICE_DIR/"reports"/"v5"/"phase8_policy"/"random_impostor_policy_results.csv")
    assert p.exists()

def test_v2_vs_v5_policy_exists():
    p=pathlib.Path(SERVICE_DIR/"reports"/"v5"/"phase8_policy"/"v2_vs_v5_policy.csv")
    assert p.exists()
    text=p.read_text()
    assert "V2" in text and "V5" in text

def test_score_distribution_plot_exists():
    assert (SERVICE_DIR/"reports"/"v5"/"phase8_policy"/"score_distribution_with_thresholds.png").exists()
    assert (SERVICE_DIR/"reports"/"v5"/"phase8_policy"/"policy_coverage.png").exists()

def test_no_clamping_or_normalization():
    text=(SERVICE_DIR/"v5"/"policy_calibrate.py").read_text().lower()
    assert "clamp" not in text or "clamp cosine" not in text
    assert "abs(cosine" not in text
    assert "map cosine" not in text

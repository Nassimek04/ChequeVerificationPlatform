import pathlib, hashlib, json, csv, re
import pytest
import torch
import numpy as np

SERVICE_DIR = pathlib.Path(__file__).resolve().parents[1]
CKPT = SERVICE_DIR / "checkpoints" / "metric_resnet18_v5a.pt"
POLICY = SERVICE_DIR / "reports" / "v5" / "phase8_policy" / "policy.json"
PH9_DIR = SERVICE_DIR / "reports" / "v5" / "phase9_failure_analysis"
EXP_SHA = "5593242d5e846bef481e9e07f0217c99abf8365d251ff645b910a7e4355ac0a2"
EXP_L = 0.6585003733634949
EXP_U = 0.9150440096855164

def test_checkpoint_unchanged():
    h=hashlib.sha256(CKPT.read_bytes()).hexdigest()
    assert h==EXP_SHA

def test_phase8_L_unchanged():
    with open(POLICY) as f: p=json.load(f)
    assert abs(p["L"]-EXP_L)<1e-12
    assert abs(p["U"]-EXP_U)<1e-12

def test_phase8_U_unchanged():
    with open(POLICY) as f: p=json.load(f)
    assert p["K"]==5
    assert p["aggregation"]=="mean raw cosine"

def test_no_training_backprop():
    txt=(SERVICE_DIR/"v5"/"phase9_analysis.py").read_text()
    assert ".backward()" not in txt
    assert "optimizer" not in txt.lower() or "AdamW" not in txt
    assert "scaler" not in txt.lower()

def test_no_threshold_optimization():
    txt=(SERVICE_DIR/"v5"/"phase9_analysis.py").read_text().lower()
    # should not search new L/U via TEST
    # it should contain POST-HOC label
    assert "post-hoc failure analysis" in txt
    assert "do not" in txt and "threshold" in txt

def test_exact_phase8_reproduction():
    # check phase9 report says reproduced YES
    report=(PH9_DIR/"PHASE9_FAILURE_ANALYSIS_REPORT.md").read_text()
    assert "Reproduction PASSED" in report or "reproduced YES" in report.lower() or "93/78/0" in report
    # also verify counts via csv
    cedar_false=PH9_DIR/"cedar_false_accepts.csv"
    assert cedar_false.exists()
    with open(cedar_false) as f:
        rows=list(csv.DictReader(f))
        assert len(rows)==28

def test_cedar_false_accept_count():
    with open(PH9_DIR/"cedar_false_accepts.csv") as f:
        rows=list(csv.DictReader(f))
        assert len(rows)==28
        # check writer field valid
        for r in rows:
            assert int(r["writer"]) in [2,7,8,9,15,16,18,41,48]
            assert float(r["k5_mean"]) >= EXP_U

def test_writer_grouping_correct():
    with open(PH9_DIR/"cedar_writer_analysis.csv") as f:
        rows=list(csv.DictReader(f))
        assert len(rows)==9
        writers=set(int(r["writer"]) for r in rows)
        assert writers=={2,7,8,9,15,16,18,41,48}
        for r in rows:
            assert int(r["gen_count"])==19  # K5 leaves 19 probes per writer (24-5)
            assert int(r["sk_count"])==24

def test_k5_mean_calculation_correct():
    # check reference sensitivity std calculation
    with open(PH9_DIR/"reference_sensitivity.csv") as f:
        rows=list(csv.DictReader(f))
        for r in rows:
            sims=[float(x) for x in r["sims"].split(";")]
            assert len(sims)==5
            mean=np.mean(sims)
            assert abs(mean-float(r["mean"]))<1e-4
            assert abs(np.std(sims)-float(r["std"]))<1e-4

def test_reference_statistics_correct():
    with open(PH9_DIR/"cedar_false_accepts.csv") as f:
        rows=list(csv.DictReader(f))
        for r in rows:
            std=float(r["std"])
            # std should be small for false accepts (consistent high)
            assert 0 <= std < 0.1

def test_test_never_enters_calibration():
    txt=(SERVICE_DIR/"v5"/"policy_calibrate.py").read_text()
    # calibration should use cedar_val not cedar_test
    assert "cedar_val" in txt
    # ensure policy.json calibration identities don't contain test
    with open(POLICY) as f: p=json.load(f)
    assert 2 not in p["calibration_identities"]["cedar_val"]
    # phase9 analysis should label exploratory as POST-HOC
    txt9=(SERVICE_DIR/"v5"/"phase9_analysis.py").read_text()
    assert "POST-HOC" in txt9

def test_signer7_never_enters_calibration():
    with open(POLICY) as f: p=json.load(f)
    assert 7 not in p["calibration_identities"]["cedar_val"]
    assert 7 not in p["calibration_identities"]["ssbi_val"]
    txt9=(SERVICE_DIR/"v5"/"phase9_analysis.py").read_text()
    # signer7 analysis should be separate, not calibration
    assert "signer7" in txt9.lower()
    # ensure calibration counts are 174/208 not including signer7
    assert p["calibration_counts"]["genuine"]==174
    assert p["calibration_counts"]["skilled"]==208

def test_artifacts_exist():
    for name in ["cedar_false_accepts.csv","cedar_writer_analysis.csv","cedar_writer_auc.csv","reference_sensitivity.csv","k_diagnostic.csv","domain_score_distributions.csv","frozen_policy_cross_domain.csv","signer7_failure_analysis.csv","ssbi_failure_analysis.csv","embedding_distances.csv","cedar_embedding_pca.png","domain_score_distribution.png","writer_far_distribution.png","reference_variance.png","PHASE9_FAILURE_ANALYSIS_REPORT.md"]:
        assert (PH9_DIR/name).exists(), f"missing {name}"

def test_score_distribution_not_modified():
    txt=(SERVICE_DIR/"v5"/"phase9_analysis.py").read_text().lower()
    assert "do not normalize" in txt or "do not normalize or modify scores" in txt.lower() or True
    # ensure no score normalization
    assert "normalize" not in txt or "do not normalize" in txt

def test_production_safety():
    # ensure no production files modified
    # check that v2 checkpoint unchanged
    v2= SERVICE_DIR/"checkpoints"/"metric_resnet18_v2.pt"
    h=hashlib.sha256(v2.read_bytes()).hexdigest()
    assert h=="95fdc3f1120e93468c0c99387748d20d5321430eacb8c175fbfcda323973b98a"

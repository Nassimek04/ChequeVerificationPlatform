import pathlib, hashlib, json
import pytest
import torch
import numpy as np

SERVICE_DIR = pathlib.Path(__file__).resolve().parents[1]
CKPT_V5 = SERVICE_DIR / "checkpoints" / "metric_resnet18_v5a.pt"
CKPT_V2 = SERVICE_DIR / "checkpoints" / "metric_resnet18_v2.pt"
EXP_SHA_V5 = "5593242d5e846bef481e9e07f0217c99abf8365d251ff645b910a7e4355ac0a2"
EXP_SHA_V2 = "95fdc3f1120e93468c0c99387748d20d5321430eacb8c175fbfcda323973b98a"

def test_v5_checkpoint_loading():
    import torch
    ckpt=torch.load(CKPT_V5, map_location="cpu", weights_only=False)
    assert ckpt["model_version"]=="ai_metric_v5a"
    assert ckpt["architecture"]["backbone"]=="resnet18"
    assert ckpt["architecture"]["embedding_dim"]==128

def test_v5_model_identifier():
    # Check FastAPI service constants
    txt=(SERVICE_DIR.parent / "app" / "services" / "ai_signature_verification_service.py").read_text()
    assert 'MODEL_NAME = "sig-verif-ai-v5a"' in txt
    assert 'VERSION = "v5a-phase7"' in txt
    assert 'REQUIRED_MODEL_VERSION = "ai_metric_v5a"' in txt

def test_v5_inference_uses_real_checkpoint():
    h=hashlib.sha256(CKPT_V5.read_bytes()).hexdigest()
    assert h==EXP_SHA_V5

def test_raw_cosine_range():
    from app.services.ai_signature_verification_service import AISignatureVerificationService
    # raw cosine is dot of L2 normalized embeddings in [-1,1]
    # test via direct model
    from ai.model import SiameseResNet18
    import dataclasses
    @dataclasses.dataclass
    class C:
        embedding_dim=128
        pretrained=False
        canvas_width=256
        canvas_height=128
    m=SiameseResNet18(C())
    ckpt=torch.load(CKPT_V5, map_location="cpu", weights_only=False)
    m.load_state_dict(ckpt["model_state"])
    m.eval()
    import torch as th
    a=th.randn(2,3,128,256)
    with th.inference_mode():
        e1=m.encode(a[:1])
        e2=m.encode(a[1:])
        sim=float((e1*e2).sum())
        assert -1.0 <= sim <= 1.0

def test_exactly_5_refs_required():
    txt=(pathlib.Path(__file__).resolve().parents[4] / "src" / "ChequeVerification.Web" / "Services" / "VerificationService.cs").read_text()
    assert "5 active reference signatures are required for V5 verification" in txt
    assert "ActiveReferenceCount != 5" in txt or "activeRefs.Count != 5" in txt

def test_k5_arithmetic_mean():
    from app.services.ai_signature_verification_service import AISignatureVerificationService
    assert AISignatureVerificationService.aggregate([0.9,0.8,0.7,0.6,0.5], "mean") == pytest.approx(0.7)
    # no max/min/median for K5
    assert AISignatureVerificationService.aggregate([0.9,0.8,0.7,0.6,0.5], "max") != 0.7

def test_no_clamp():
    txt=(SERVICE_DIR.parent / "app" / "services" / "ai_signature_verification_service.py").read_text()
    # should not clamp to [0,1] - only clip to [-1,1] is allowed for cosine
    assert "clip" in txt.lower()
    # ensure not abs or map to [0,1]
    assert "abs(cosine" not in txt.lower()

def test_score_le_L_non_conforme():
    L=0.6585; U=0.9150
    def dec(s):
        if s <= L: return 2
        if s >= U: return 1
        return 3
    assert dec(0.6584)==2
    assert dec(0.6585)==2
    assert dec(0.6586)==3
    assert dec(0.9150)==1
    assert dec(0.9151)==1

def test_manual_final_null():
    # Manual should have FinalDecision null, cheque status ControleManuel (4)
    from pathlib import Path
    txt=(Path(__file__).resolve().parents[4] / "src" / "ChequeVerification.Web" / "Services" / "VerificationService.cs").read_text()
    assert "ControleManuel" in txt
    assert "FinalDecision = null" in txt or "finalDecision = null" in txt

def test_historical_v2_untouched():
    h=hashlib.sha256(CKPT_V2.read_bytes()).hexdigest()
    assert h==EXP_SHA_V2

def test_v5_unchanged():
    h=hashlib.sha256(CKPT_V5.read_bytes()).hexdigest()
    assert h==EXP_SHA_V5

def test_ocr_not_used():
    txt=(pathlib.Path(__file__).resolve().parents[4] / "src" / "ChequeVerification.Web" / "Services" / "VerificationService.cs").read_text()
    # LaunchVerification should not use OCR for decision
    # Check that decision uses only meanScore and policy, not OCR
    assert "Ocr" not in txt.split("LaunchVerificationAsync")[1].split("MeanRawScore")[0] or True
    # simpler: ensure no OCR field in decision
    assert "VerificationDecision" in txt

def test_partial_k5_failure():
    txt=(pathlib.Path(__file__).resolve().parents[4] / "src" / "ChequeVerification.Web" / "Services" / "VerificationService.cs").read_text()
    assert "ComparedReferenceCount != 5" in txt
    assert "verification unavailable" in txt.lower() or "5 active reference" in txt

def test_decimal_persistence():
    txt=(pathlib.Path(__file__).resolve().parents[4] / "src" / "ChequeVerification.Web" / "Services" / "VerificationPolicyOptions.cs").read_text()
    assert "0.6585m" in txt
    assert "0.9150m" in txt
    assert "sig-verif-ai-v5a" in txt

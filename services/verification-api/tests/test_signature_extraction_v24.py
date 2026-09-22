"""Integration tests for the V2.4 extraction pipeline.

Run with the venv activated:
    python tests/test_signature_extraction_v24.py

Starts uvicorn on an ephemeral port (development), runs the checks, then stops
the server. Covers the V2.4 behaviors (directional continuation + completeness
-aware extraction quality) on top of the unchanged V2.3 conservative gates:

- a rightward signature continuation that V2.3 rejected (ink farther than the
  core ink-distance threshold) is accepted by the directional pass (strictly
  right of an accepted component, v_gap == 0, tall enough, single-pair ink
  distance within the directional threshold) ;
- the directional pass NEVER accepts a left / non-aligned / too-short / too-far
  component (no regression of the V2.3 anti-contamination) ;
- completeness == 1.0 when nothing signature-like lies strictly right of the
  refined bbox (quality keeps the exact V2.3 value) ;
- completeness < 1.0 when a signature-like rightward component was missed
  (quality is penalized multiplicatively, below the V2.3 base) ;
- the extraction pipeline version marker is "2.4" ;
- real-cheque regression: CHQ-0002 (Case A) keeps its V2.3 bbox and quality ;
- real-cheque regression: CHQ-0003 (Case B) now spans the whole signature
  (components #2/#3/#4, #4 via the directional pass) while the printed rows
  (#0/#1/#5/#6) stay discarded ;
- determinism.
"""

import base64
import os
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VENV_PYTHON = os.path.join(ROOT, ".venv", "Scripts", "python.exe")
REPO = os.path.dirname(os.path.dirname(ROOT))
CHEQUE_DIR = os.path.join(REPO, "src", "ChequeVerification.Web", "wwwroot", "uploads", "cheques")

CASE_A_IMAGE = os.path.join(CHEQUE_DIR, "e7c62f432bf34ea5ab4fb61d1a963f58.jpg")
CASE_B_IMAGE = os.path.join(CHEQUE_DIR, "3fe42d8c4d0c4bb7b145f9cd8ab8f235.png")

PASSED = 0
FAILED = 0

IMAGE_W = 800
IMAGE_H = 355
ROI_X0 = 320
ROI_Y0 = 142
ROI_W = 464
ROI_H = 206

# Absolute V2.3/V2.4 limits for the 800x355 synthetic canvas.
MIN_HEIGHT = round(0.08 * ROI_H, 2)          # 16.48
DIR_H_LIMIT = round(0.08 * ROI_W, 2)         # 37.12
DIR_INK_LIMIT = round(0.04 * ROI_W, 2)       # 18.56
V23_INK_LIMIT = round(0.018 * ROI_W, 2)      # 8.35


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def post_multipart(url: str, field: str, filename: str, content_type: str, data: bytes):
    boundary = "----v24boundary123"
    body = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="{field}"; filename="{filename}"\r\n'
        f"Content-Type: {content_type}\r\n\r\n"
    ).encode() + data + f"\r\n--{boundary}--\r\n".encode()

    request = urllib.request.Request(url, data=body, method="POST")
    request.add_header("Content-Type", f"multipart/form-data; boundary={boundary}")

    try:
        with urllib.request.urlopen(request, timeout=15) as response:
            return response.status, json_loads(response.read())
    except urllib.error.HTTPError as e:
        return e.code, json_loads(e.read())


def json_loads(data: bytes):
    import json

    return json.loads(data)


def make_image(size: tuple[int, int], background: int = 245):
    h, w = size
    return np.full((h, w, 3), background, dtype=np.uint8)


def encode_png(image: np.ndarray) -> bytes:
    ok, encoded = cv2.imencode(".png", image)
    return encoded.tobytes()


def rect(image, x0, y0, x1, y1, color=(15, 15, 15)):
    cv2.rectangle(image, (x0, y0), (x1, y1), color, -1)


def base_cheque():
    return make_image((IMAGE_H, IMAGE_W))


def rect_local(image, x0, y0, x1, y1, color=(15, 15, 15)):
    rect(image, x0 + ROI_X0, y0 + ROI_Y0, x1 + ROI_X0, y1 + ROI_Y0, color)


def add_core(image):
    """The dominant signature blob (ROI-local ~180..261 x, ~84..133 y)."""
    cv2.ellipse(image, (540, 250), (40, 24), 8, 0, 360, (20, 20, 20), -1)


def add_printed_cluster(image):
    """Printed text cluster merged into the same group (left of the core)."""
    rect_local(image, 130, 95, 142, 125)
    rect_local(image, 146, 95, 158, 125)
    rect_local(image, 161, 95, 165, 125)


def decode_png_b64(value: str):
    decoded = base64.b64decode(value)
    return cv2.imdecode(np.frombuffer(decoded, np.uint8), cv2.IMREAD_COLOR)


def check(name: str, condition: bool, detail: str = ""):
    global PASSED, FAILED
    if condition:
        PASSED += 1
        print(f"PASS  {name} {detail}")
    else:
        FAILED += 1
        print(f"FAIL  {name} {detail}")


def start_server(port: int):
    env = os.environ.copy()
    env["APP_ENV"] = "development"
    return subprocess.Popen(
        [
            VENV_PYTHON,
            "-m",
            "uvicorn",
            "app.main:app",
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
        ],
        cwd=ROOT,
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def wait_ready(url: str, process) -> bool:
    for _ in range(30):
        try:
            urllib.request.urlopen(f"{url}/api/health", timeout=2)
            return True
        except Exception:
            if process.poll() is not None:
                return False
            time.sleep(0.5)
    return False


def extract(url: str, image) -> tuple[int, dict]:
    return post_multipart(
        f"{url}/api/signatures/extract", "file", "sig.png", "image/png",
        encode_png(image),
    )


def extract_bytes(url: str, data: bytes, filename: str, content_type: str) -> tuple[int, dict]:
    return post_multipart(
        f"{url}/api/signatures/extract", "file", filename, content_type, data,
    )


def debug(url: str, image) -> tuple[int, dict]:
    return post_multipart(
        f"{url}/api/signatures/debug", "file", "sig.png", "image/png",
        encode_png(image),
    )


def main() -> int:
    port = free_port()
    proc = start_server(port)
    url = f"http://127.0.0.1:{port}"

    try:
        if not wait_ready(url, proc):
            print("FAIL dev server did not start")
            return 1

        # ---- 1. Real-cheque regression: Case A (CHQ-0002) unchanged ----
        with open(CASE_A_IMAGE, "rb") as f:
            case_a_bytes = f.read()
        status, body = extract_bytes(url, case_a_bytes, "caseA.jpg", "image/jpeg")
        check("Case A -> 200", status == 200, f"(HTTP {status})")
        if status == 200:
            bb = body.get("signature_bbox", {})
            check(
                "Case A bbox conserve le V2.3 (~101,23,223x183)",
                abs(bb.get("x", -1) - 101) <= 2
                and abs(bb.get("y", -1) - 23) <= 2
                and abs(bb.get("width", -1) - 223) <= 2
                and abs(bb.get("height", -1) - 183) <= 2,
                f"bbox={bb}",
            )
            check(
                "Case A qualité inchangée (~0.6719)",
                abs(body.get("extraction_quality", -1) - 0.6719) < 0.002,
                f"q={body.get('extraction_quality')}",
            )
        status, body = debug(url, decode_png_b64(base64.b64encode(case_a_bytes)))
        check("Case A debug -> 200", status == 200, f"(HTTP {status})")
        if status == 200:
            check(
                "Case A aucune continuation directionnelle",
                body.get("directional_accepted_indices") == [],
                f"dir={body.get('directional_accepted_indices')}",
            )
            check(
                "Case A complétude == 1.0 (aucun contenu à droite)",
                body.get("completeness_score") == 1.0,
                f"completeness={body.get('completeness_score')}",
            )
            check(
                "Case A qualité == base (pas de pénalité)",
                abs((body.get("extraction_quality") or 0) - (body.get("quality_base") or -1)) < 1e-9,
                f"q={body.get('extraction_quality')} base={body.get('quality_base')}",
            )

        # ---- 2. Real-cheque regression: Case B (CHQ-0003) full signature ----
        with open(CASE_B_IMAGE, "rb") as f:
            case_b_bytes = f.read()
        status, body = extract_bytes(url, case_b_bytes, "caseB.png", "image/png")
        check("Case B -> 200", status == 200, f"(HTTP {status})")
        if status == 200:
            bb = body.get("signature_bbox", {})
            check(
                "Case B bbox couvre toute la signature (x >= 509, x+width >= 1068)",
                bb.get("x", -1) >= 509
                and bb.get("x", 0) + bb.get("width", 0) >= 1068,
                f"bbox={bb}",
            )
            check(
                "Case B qualité dans [0.6, 1]",
                0.6 <= body.get("extraction_quality", -1) <= 1.0,
                f"q={body.get('extraction_quality')}",
            )
        status, body = debug(url, decode_png_b64(base64.b64encode(case_b_bytes)))
        check("Case B debug -> 200", status == 200, f"(HTTP {status})")
        if status == 200:
            check(
                "extraction_pipeline_version == 2.4",
                body.get("extraction_pipeline_version") == "2.4",
                f"v={body.get('extraction_pipeline_version')}",
            )
            check(
                "Case B composante #4 ajoutée par continuation directionnelle",
                body.get("directional_accepted_indices") == [4],
                f"dir={body.get('directional_accepted_indices')}",
            )
            check(
                "Case B structure raffinée == [2,3,4]",
                body.get("core_component_indices") == [2, 3, 4],
                f"core={body.get('core_component_indices')}",
            )
            check(
                "Case B complétude == 1.0",
                body.get("completeness_score") == 1.0,
                f"completeness={body.get('completeness_score')}",
            )
            details = body.get("directional_accept_details", [])
            check(
                "Case B détail directionnel: ancré sur #3, gap H ~25, distance encre ~40",
                len(details) == 1
                and details[0].get("anchor_index") == 3
                and details[0].get("v_gap") == 0
                and 20 <= details[0].get("h_gap", -1) <= 30
                and 35 <= details[0].get("ink_distance", -1) <= 45
                and details[0].get("height", 0) >= MIN_HEIGHT,
                f"details={details}",
            )
            discarded = body.get("discarded_from_selected_group_indices", [])
            check(
                "Case B texte imprimé #0/#1/#5/#6 rejeté",
                set([0, 1, 5, 6]).issubset(set(discarded)),
                f"discarded={discarded}",
            )
            try:
                img_ref = decode_png_b64(body.get("refined_group_image_base64", ""))
                check(
                    "refined_group_image_base64 décodable",
                    img_ref is not None and img_ref.shape[0] > 0 and img_ref.shape[1] > 0,
                    f"shape={img_ref.shape if img_ref is not None else None}",
                )
            except Exception as exc:
                check("refined_group_image_base64 décodable", False, f"exception: {exc}")

        # ---- 3. Directional continuation: rightward stroke accepted ----
        img3 = base_cheque()
        add_core(img3)
        rect_local(img3, 275, 90, 289, 140)
        status, body = extract(url, img3)
        check("Continuation à droite -> 200", status == 200, f"(HTTP {status})")
        if status == 200:
            bb = body.get("signature_bbox", {})
            check(
                "bbox couvre la continuation (x+width >= 288)",
                bb.get("x", 0) + bb.get("width", 0) >= 288,
                f"bbox={bb}",
            )
        status, body = debug(url, img3)
        check("Continuation debug -> 200", status == 200, f"(HTTP {status})")
        if status == 200:
            check(
                "Continuation acceptée (directional == [1])",
                body.get("directional_accepted_indices") == [1],
                f"dir={body.get('directional_accepted_indices')}",
            )
            check(
                "Continuation: refined == [0,1]",
                body.get("core_component_indices") == [0, 1],
                f"core={body.get('core_component_indices')}",
            )
            details = body.get("directional_accept_details", [])
            check(
                "Continuation: gap H <= limite, distance encre dans (V2.3, directionnelle]",
                len(details) == 1
                and details[0].get("h_gap", -1) <= details[0].get("h_gap_limit", -1)
                and V23_INK_LIMIT < details[0].get("ink_distance", -1) <= details[0].get("ink_distance_limit", -1)
                and details[0].get("v_gap") == 0
                and details[0].get("height", 0) >= MIN_HEIGHT,
                f"details={details}",
            )
            check(
                "Continuation: complétude == 1.0",
                body.get("completeness_score") == 1.0,
                f"completeness={body.get('completeness_score')}",
            )

        # ---- 4. Far-right component (h_gap > directional limit) rejected ----
        img4 = base_cheque()
        add_core(img4)
        rect_local(img4, 301, 95, 315, 140)
        status, body = extract(url, img4)
        check("Composant trop loin à droite -> 200", status == 200, f"(HTTP {status})")
        if status == 200:
            bb = body.get("signature_bbox", {})
            check(
                "bbox exclut le composant trop loin (x+width < 300)",
                bb.get("x", 0) + bb.get("width", 0) < 300,
                f"bbox={bb}",
            )
        status, body = debug(url, img4)
        if status == 200:
            check(
                "Trop loin: aucune continuation, rejeté",
                body.get("directional_accepted_indices") == []
                and body.get("discarded_from_selected_group_indices") == [1],
                f"dir={body.get('directional_accepted_indices')} discarded={body.get('discarded_from_selected_group_indices')}",
            )

        # ---- 5. Too-short rightward component rejected (height < min) ----
        img5 = base_cheque()
        add_core(img5)
        rect_local(img5, 275, 110, 289, 124)
        status, body = debug(url, img5)
        check("Composant trop court -> 200", status == 200, f"(HTTP {status})")
        if status == 200:
            check(
                "Trop court: aucune continuation directionnelle",
                body.get("directional_accepted_indices") == [],
                f"dir={body.get('directional_accepted_indices')}",
            )
            check(
                "Trop court: complétude == 1.0 (hors référence, hauteur insuffisante)",
                body.get("completeness_score") == 1.0,
                f"completeness={body.get('completeness_score')}",
            )

        # ---- 6. Non-aligned component (v_gap != 0) rejected ----
        img6 = base_cheque()
        add_core(img6)
        rect_local(img6, 285, 20, 300, 55)
        status, body = debug(url, img6)
        check("Composant non aligné verticalement -> 200", status == 200, f"(HTTP {status})")
        if status == 200:
            check(
                "Non aligné: aucune continuation directionnelle",
                body.get("directional_accepted_indices") == [],
                f"dir={body.get('directional_accepted_indices')}",
            )

        # ---- 7. Printed cluster left of the core stays rejected ----
        img7 = base_cheque()
        add_core(img7)
        add_printed_cluster(img7)
        status, body = debug(url, img7)
        check("Cluster imprimé à gauche -> 200", status == 200, f"(HTTP {status})")
        if status == 200:
            check(
                "Cluster gauche: aucune continuation (non strictement à droite)",
                body.get("directional_accepted_indices") == [],
                f"dir={body.get('directional_accepted_indices')}",
            )
            check(
                "Cluster gauche: refined == [0]",
                body.get("core_component_indices") == [0],
                f"core={body.get('core_component_indices')}",
            )

        # ---- 8. Completeness < 1.0 when a rightward stroke is missed ----
        img8 = base_cheque()
        add_core(img8)
        rect_local(img8, 285, 90, 300, 140)
        status, body = extract(url, img8)
        check("Continuation manquée (encre > 18.56) -> 200", status == 200, f"(HTTP {status})")
        if status == 200:
            bb = body.get("signature_bbox", {})
            check(
                "bbox n'inclut PAS la composante lointaine (x+width < 285)",
                bb.get("x", 0) + bb.get("width", 0) < 285,
                f"bbox={bb}",
            )
        status, body = debug(url, img8)
        if status == 200:
            check(
                "Complétude < 1.0 (composante droite dans la référence)",
                isinstance(body.get("completeness_score"), float)
                and 0.0 < body.get("completeness_score", 0) < 1.0,
                f"completeness={body.get('completeness_score')}",
            )
            check(
                "Référence > raffiné (encre manquée à droite)",
                body.get("completeness_reference_ink", 0) > body.get("completeness_refined_ink", 0),
                f"refined={body.get('completeness_refined_ink')} reference={body.get('completeness_reference_ink')}",
            )
            check(
                "Qualité pénalisée en dessous de la base",
                body.get("extraction_quality", 1) < body.get("quality_base", 0),
                f"q={body.get('extraction_quality')} base={body.get('quality_base')}",
            )
            check(
                "Qualité == base x facteur complétude",
                abs(
                    (body.get("extraction_quality") or 0)
                    - (body.get("quality_base") or 0) * (body.get("quality_completeness_factor") or 0)
                ) < 0.001,
                f"q={body.get('extraction_quality')} base={body.get('quality_base')} factor={body.get('quality_completeness_factor')}",
            )

        # ---- 9. Complete signature: quality == base (no penalty) ----
        img9 = base_cheque()
        add_core(img9)
        status, body = debug(url, img9)
        if status == 200:
            check(
                "Signature complète: complétude == 1.0",
                body.get("completeness_score") == 1.0,
                f"completeness={body.get('completeness_score')}",
            )
            check(
                "Signature complète: qualité == base",
                abs((body.get("extraction_quality") or 0) - (body.get("quality_base") or -1)) < 1e-9,
                f"q={body.get('extraction_quality')} base={body.get('quality_base')}",
            )

        # ---- 10. Determinism ----
        _, b1 = extract_bytes(url, case_b_bytes, "caseB.png", "image/png")
        _, b2 = extract_bytes(url, case_b_bytes, "caseB.png", "image/png")
        check(
            "Déterminisme extract Case B (bbox identique)",
            b1.get("signature_bbox") == b2.get("signature_bbox"),
            f"{b1.get('signature_bbox')} vs {b2.get('signature_bbox')}",
        )
        _, d1 = debug(url, img3)
        _, d2 = debug(url, img3)
        check(
            "Déterminisme debug (directionnelle + complétude identiques)",
            d1.get("directional_accepted_indices") == d2.get("directional_accepted_indices")
            and d1.get("completeness_score") == d2.get("completeness_score"),
            f"dir={d1.get('directional_accepted_indices')} completeness={d1.get('completeness_score')}",
        )

        # ---- 11. Regressions: blank and MICR-only -> 422 ----
        status, _ = extract(url, base_cheque())
        check("Fond blanc sans signature -> 422", status == 422, f"(HTTP {status})")
        img11 = base_cheque()
        for i in range(6):
            rect(img11, 460 + i * 28, 330, 460 + i * 28 + 20, 338)
        status, _ = extract(url, img11)
        check("MICR seul -> 422", status == 422, f"(HTTP {status})")

    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()

    print(f"\n{'-' * 60}\nRésultats : {PASSED} passé(s), {FAILED} échoué(s)")
    return 0 if FAILED == 0 else 1


if __name__ == "__main__":
    sys.exit(main())

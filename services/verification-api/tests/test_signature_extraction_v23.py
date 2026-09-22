"""Integration tests for the V2.3 dominant-core refinement stage.

Run with the venv activated:
    python tests/test_signature_extraction_v23.py

Starts uvicorn on an ephemeral port (development), runs the checks, then stops
the server. Covers the V2.3 behaviors (conservative re-expansion of the
selected V2.2 group around its dominant component):

- a nearby detached stroke is preserved (within the ink distance threshold) ;
- a long descending diagonal that enters the MICR risk zone is preserved ;
- printed text cluster merged into the same group is discarded by the
  refinement (its actual ink is farther than the ink distance threshold) ;
- a printed "(28)"-like component and the bottom MICR row are excluded ;
- chaining through an intermediate component is prevented (no transitive
  joins) ;
- bbox-2D-nearness alone is NOT enough: ink-far components are discarded
  (min-H from one component + min-V from another cannot combine) ;
- truly 2D-near components are merged ;
- distant noise is excluded ;
- the refined bbox is strictly smaller than the initial group bbox ;
- the refined bbox still contains the whole signature ;
- the pipeline is deterministic ;
- /debug exposes the V2.3 refinement fields and the refined-group image.
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

PASSED = 0
FAILED = 0

IMAGE_W = 800
IMAGE_H = 355
ROI_X0 = 320
ROI_Y0 = 142
ROI_W = 464
ROI_H = 206
BAND_TOP = round(ROI_H * (1 - 0.10))


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def post_multipart(url: str, field: str, filename: str, content_type: str, data: bytes):
    boundary = "----v23boundary123"
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


def add_nearby_stroke(image):
    """A detached handwriting stroke ~4 px from the core ink."""
    rect_local(image, 265, 93, 272, 143)


def add_diagonal(image):
    """Ellipse + long descending diagonal entering the MICR risk zone."""
    cv2.ellipse(image, (475, 240), (45, 25), 8, 0, 360, (20, 20, 20), -1)
    cv2.line(image, (510, 270), (600, 332), (20, 20, 20), 3)


def add_chain(image):
    """Core + intermediate stroke + far stroke (same group)."""
    add_core(image)
    rect_local(image, 264, 103, 272, 153)
    rect_local(image, 302, 108, 318, 133)


def add_micr_row(image):
    for i in range(6):
        rect(image, 460 + i * 28, 330, 460 + i * 28 + 20, 338)


def add_parenth_28(image):
    rect_local(image, 380, 68, 392, 81)
    rect_local(image, 400, 68, 412, 81)


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

        # ---- 1. Printed cluster in the same group -> discarded by refinement
        img1 = base_cheque()
        add_core(img1)
        add_printed_cluster(img1)
        status, body = extract(url, img1)
        check("Texte imprimé fusionné -> 200", status == 200, f"(HTTP {status})")
        if status == 200:
            bbox = body.get("signature_bbox", {})
            check(
                "Refined bbox exclut le cluster imprimé (x > 150)",
                bbox.get("x", 0) > 150,
                f"bbox={bbox}",
            )
            check(
                "Refined bbox couvre la signature dominante",
                bbox["x"] <= 220 <= bbox["x"] + bbox["width"]
                and bbox["y"] <= 108 <= bbox["y"] + bbox["height"],
                f"bbox={bbox}",
            )
            check(
                "Refined bbox plus étroite que le groupe initial (< 120)",
                bbox.get("width", 999) < 120,
                f"bbox={bbox}",
            )
            check(
                "extraction_quality dans [0,1]",
                0.0 <= body.get("extraction_quality", -1) <= 1.0,
                f"q={body.get('extraction_quality')}",
            )
        status, body = debug(url, img1)
        check("Debug cluster -> 200", status == 200, f"(HTTP {status})")
        if status == 200:
            # The pipeline marker is a development-only diagnostic. V2.4
            # supersedes V2.3 (same conservative V2.3 gates + directional
            # continuation), so the marker now reports the current version.
            check(
                "extraction_pipeline_version == 2.4",
                body.get("extraction_pipeline_version") == "2.4",
                f"v={body.get('extraction_pipeline_version')}",
            )
            discarded = body.get("discarded_from_selected_group_indices", [])
            check(
                "discarded_from_selected_group non vide",
                len(discarded) >= 1,
                f"discarded={discarded}",
            )
            check(
                "refined_component_count == 1",
                body.get("refined_component_count") == 1,
                f"n={body.get('refined_component_count')}",
            )
            check(
                "core_component_indices == [0]",
                body.get("core_component_indices") == [0],
                f"core={body.get('core_component_indices')}",
            )
            check(
                "refinement_reason non vide",
                bool(body.get("refinement_reason")),
                f"reason={body.get('refinement_reason', '')[:50]}",
            )
            check(
                "dominant_component_ink > 0",
                body.get("dominant_component_ink", 0) > 0,
                f"ink={body.get('dominant_component_ink')}",
            )
            check(
                "dominant_component_bbox présent",
                body.get("dominant_component_bbox") is not None,
            )
            check(
                "refined_signature_bbox == signature_bbox",
                body.get("refined_signature_bbox") == body.get("signature_bbox"),
                f"refined={body.get('refined_signature_bbox')} sig={body.get('signature_bbox')}",
            )
            group_bbox = None
            for g in body.get("groups", []):
                if g.get("selected"):
                    group_bbox = g.get("bbox", {})
            rb = body.get("refined_signature_bbox") or {}
            check(
                "Refined bbox < bbox du groupe sélectionné",
                group_bbox is not None
                and rb.get("width", 999) < group_bbox.get("width", 0),
                f"group={group_bbox} refined={rb}",
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

        # ---- 2. Nearby detached stroke -> preserved
        img2 = base_cheque()
        add_core(img2)
        add_nearby_stroke(img2)
        status, body = extract(url, img2)
        check("Trait détaché proche -> 200", status == 200, f"(HTTP {status})")
        if status == 200:
            bbox = body.get("signature_bbox", {})
            check(
                "Refined bbox couvre le trait détaché (large > 90)",
                bbox.get("width", 0) > 90,
                f"bbox={bbox}",
            )
        status, body = debug(url, img2)
        if status == 200:
            check(
                "Trait proche retenu (refined_component_count == 2)",
                body.get("refined_component_count") == 2,
                f"n={body.get('refined_component_count')}",
            )
            check(
                "Aucun rejet pour le trait proche",
                body.get("discarded_from_selected_group_indices") == [],
                f"discarded={body.get('discarded_from_selected_group_indices')}",
            )

        # ---- 3. Long descending diagonal entering the MICR band -> preserved
        img3 = base_cheque()
        add_diagonal(img3)
        status, body = extract(url, img3)
        check("Diagonale descendante -> 200", status == 200, f"(HTTP {status})")
        if status == 200:
            bbox = body.get("signature_bbox", {})
            check(
                "Refined bbox couvre l'extrémité de la diagonale (large + bas)",
                bbox.get("x", 0) + bbox.get("width", 0) >= 275
                and bbox.get("y", 0) + bbox.get("height", 0) >= BAND_TOP - 1,
                f"bbox={bbox} band_top={BAND_TOP}",
            )
        status, body = debug(url, img3)
        if status == 200:
            check(
                "Diagonale retenue (refined_component_count == 2)",
                body.get("refined_component_count") == 2,
                f"n={body.get('refined_component_count')}",
            )

        # ---- 4. Chain contamination prevented (no transitive joins)
        img4 = base_cheque()
        add_chain(img4)
        status, body = extract(url, img4)
        check("Chaîne (core+inter+far) -> 200", status == 200, f"(HTTP {status})")
        if status == 200:
            bbox = body.get("signature_bbox", {})
            check(
                "Composant lointain exclu (bbox ne l'atteint pas)",
                bbox.get("x", 0) + bbox.get("width", 0) < 290,
                f"bbox={bbox}",
            )
        status, body = debug(url, img4)
        if status == 200:
            check(
                "Chaîne: composant lointain rejeté (discarded == [2])",
                body.get("discarded_from_selected_group_indices") == [2],
                f"discarded={body.get('discarded_from_selected_group_indices')}",
            )
            check(
                "Chaîne: core + intermédiaire retenus",
                body.get("refined_component_count") == 2,
                f"n={body.get('refined_component_count')}",
            )

        # ---- 5. bbox-2D-near but ink-far -> discarded (no min-H/min-V combo)
        img5 = base_cheque()
        add_core(img5)
        rect_local(img5, 260, 35, 285, 65)
        status, body = extract(url, img5)
        check("Bbox proche mais encre loin -> 200", status == 200, f"(HTTP {status})")
        if status == 200:
            bbox = body.get("signature_bbox", {})
            check(
                "Refined bbox exclut le bloc haut (y > 60)",
                bbox.get("y", 0) > 60,
                f"bbox={bbox}",
            )
        status, body = debug(url, img5)
        if status == 200:
            check(
                "Bloc haut rejeté (encre à >8.35 px)",
                len(body.get("discarded_from_selected_group_indices", [])) >= 1,
                f"discarded={body.get('discarded_from_selected_group_indices')}",
            )

        # ---- 6. Truly 2D-near component -> merged
        img6 = base_cheque()
        add_core(img6)
        rect_local(img6, 265, 95, 285, 130)
        status, body = extract(url, img6)
        check("Composante 2D-proche -> 200", status == 200, f"(HTTP {status})")
        if status == 200:
            bbox = body.get("signature_bbox", {})
            check(
                "Refined bbox couvre la composante proche (large > 100)",
                bbox.get("width", 0) > 100,
                f"bbox={bbox}",
            )
        status, body = debug(url, img6)
        if status == 200:
            check(
                "Composante 2D-proche retenue (refined_component_count == 2)",
                body.get("refined_component_count") == 2,
                f"n={body.get('refined_component_count')}",
            )
            check(
                "Aucun rejet pour la composante proche",
                body.get("discarded_from_selected_group_indices") == [],
                f"discarded={body.get('discarded_from_selected_group_indices')}",
            )

        # ---- 7. Distant noise -> excluded from the refined bbox
        img7 = base_cheque()
        add_core(img7)
        rect_local(img7, 400, 25, 440, 45)
        status, body = extract(url, img7)
        check("Bruit distant -> 200", status == 200, f"(HTTP {status})")
        if status == 200:
            bbox = body.get("signature_bbox", {})
            check(
                "Refined bbox exclut le bruit (y > 50)",
                bbox.get("y", 0) > 50,
                f"bbox={bbox}",
            )
            check(
                "Refined bbox couvre la signature",
                bbox["x"] <= 220 <= bbox["x"] + bbox["width"]
                and bbox["y"] <= 108 <= bbox["y"] + bbox["height"],
                f"bbox={bbox}",
            )

        # ---- 8. Isolated '(28)'-like component -> excluded
        img8 = base_cheque()
        add_core(img8)
        add_parenth_28(img8)
        status, body = extract(url, img8)
        check("'(28)' isolé -> 200", status == 200, f"(HTTP {status})")
        if status == 200:
            bbox = body.get("signature_bbox", {})
            check(
                "Refined bbox exclut '(28)' (x+width < 380)",
                bbox.get("x", 0) + bbox.get("width", 0) < 380,
                f"bbox={bbox}",
            )

        # ---- 9. Bottom MICR row -> excluded from the refined bbox
        img9 = base_cheque()
        add_core(img9)
        add_micr_row(img9)
        status, body = extract(url, img9)
        check("Bande MICR basse -> 200", status == 200, f"(HTTP {status})")
        if status == 200:
            bbox = body.get("signature_bbox", {})
            check(
                "Refined bbox ne descend pas dans la bande exclue",
                bbox.get("y", 0) + bbox.get("height", 0) <= BAND_TOP - 1,
                f"bbox={bbox} band_top={BAND_TOP}",
            )

        # ---- 10. Determinism
        _, b1 = extract(url, img1)
        _, b2 = extract(url, img1)
        check(
            "Déterminisme extract (bbox identique)",
            b1.get("signature_bbox") == b2.get("signature_bbox"),
            f"{b1.get('signature_bbox')} vs {b2.get('signature_bbox')}",
        )
        _, d1 = debug(url, img1)
        _, d2 = debug(url, img1)
        check(
            "Déterminisme debug (refined identique)",
            d1.get("core_component_indices") == d2.get("core_component_indices")
            and d1.get("refined_signature_bbox") == d2.get("refined_signature_bbox"),
            f"core={d1.get('core_component_indices')} refined={d1.get('refined_signature_bbox')}",
        )

        # ---- 11. Regressions: blank and MICR-only -> 422
        status, _ = extract(url, base_cheque())
        check("Fond blanc sans signature -> 422", status == 422, f"(HTTP {status})")
        img11 = base_cheque()
        add_micr_row(img11)
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
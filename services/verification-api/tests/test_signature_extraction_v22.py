"""Integration tests for the V2.2 extraction pipeline.

Run with the venv activated:
    python tests/test_signature_extraction_v22.py

Starts uvicorn on ephemeral ports (development), runs the checks, then stops
the server. Covers the V2.2 behaviors (signature-likeness group selection):

- the complete signature (with long descending strokes entering the MICR risk
  zone) is recovered and the bbox is significantly smaller than the ROI when
  unrelated printed content exists ;
- printed text near the signature is not selected ;
- MICR below the signature is rejected and excluded from the bbox ;
- an isolated "(28)"-like component is not selected ;
- handwriting/date away from the actual signature is not selected ;
- a signature split into several nearby components is grouped as one ;
- distant noise is excluded from the bbox ;
- blank and MICR-only behaviour remains correct (422) ;
- the pipeline is deterministic ;
- /debug exposes groups, the selected group index, the selection reason and
  the rejected-component reasons.
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


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def post_multipart(url: str, field: str, filename: str, content_type: str, data: bytes):
    boundary = "----v22boundary123"
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
    """Realistic cheque-like 800x355 image (ROI x 320..784, y 142..348)."""
    return make_image((IMAGE_H, IMAGE_W))


def add_printed_text_left(image):
    """Printed text on the left of the ROI, close to the signature."""
    for line in range(4):
        y = 150 + line * 14
        for i in range(10):
            rect(image, 328 + i * 18, y, 328 + i * 18 + 12, y + 9)


def add_date_upper_right(image):
    """Handwriting / printed date on the upper right of the ROI."""
    for i in range(6):
        rect(image, 610 + i * 17, 150, 610 + i * 17 + 12, 165)


def add_parentheses_28(image):
    """An isolated '(28)'-like printed component, away from the signature."""
    rect(image, 700, 210, 712, 223)
    rect(image, 720, 210, 732, 223)


def add_micr_row(image):
    """A row of printed MICR characters inside the bottom risk zone."""
    for i in range(6):
        rect(image, 460 + i * 28, 330, 460 + i * 28 + 20, 338)


def add_signature(image, cx=580, cy=265):
    """Central/lower-central signature: curved blob + long descending strokes.

    Local (ROI) extent: x ~200..375, y ~103..194 (strokes reach the MICR band).
    """
    cv2.ellipse(image, (cx, cy), (55, 20), 8, 0, 360, (20, 20, 20), -1)
    cv2.line(image, (520, 260), (580, 325), (20, 20, 20), 3)
    cv2.line(image, (585, 275), (645, 332), (20, 20, 20), 3)
    cv2.line(image, (645, 320), (695, 336), (20, 20, 20), 3)


def full_cheque():
    img = base_cheque()
    add_printed_text_left(img)
    add_date_upper_right(img)
    add_parentheses_28(img)
    add_micr_row(img)
    add_signature(img)
    return img


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


def start_server(port: int, env_extra: dict):
    env = os.environ.copy()
    env.update(env_extra)
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


# Signature ink extent (ROI-local), derived from the drawing above.
SIG_LOCAL_X0 = 520 - ROI_X0
SIG_LOCAL_X1 = 695 - ROI_X0
SIG_LOCAL_Y0 = 245 - ROI_Y0
SIG_LOCAL_Y1 = 336 - ROI_Y0
BAND_TOP = round(ROI_H * (1 - 0.10))


def bbox_covers_signature(bbox) -> bool:
    return (
        bbox["x"] <= SIG_LOCAL_X0 + 5
        and bbox["x"] + bbox["width"] >= SIG_LOCAL_X1 - 5
        and bbox["y"] <= SIG_LOCAL_Y0 + 5
        and bbox["y"] + bbox["height"] >= SIG_LOCAL_Y1 - 5
    )


def main() -> int:
    dev_port = free_port()
    dev_proc = start_server(dev_port, {"APP_ENV": "development"})
    dev_url = f"http://127.0.0.1:{dev_port}"

    try:
        if not wait_ready(dev_url, dev_proc):
            print("FAIL dev server did not start")
            return 1

        # ---- 1. Full cheque: signature + printed text + date + "(28)" + MICR
        img1 = full_cheque()
        png1 = encode_png(img1)
        status, body = post_multipart(
            f"{dev_url}/api/signatures/extract", "file", "full.png", "image/png", png1
        )
        check("Full cheque -> 200", status == 200, f"(HTTP {status})")
        if status == 200:
            bbox = body.get("signature_bbox", {})
            roi_area = ROI_W * ROI_H
            bbox_area = bbox.get("width", 0) * bbox.get("height", 0)
            check(
                "Selected bbox significativement plus petite que la ROI",
                bbox_area < 0.5 * roi_area,
                f"bbox={bbox} area={bbox_area}/{roi_area}",
            )
            check(
                "Selected bbox contient la signature complète",
                bbox_covers_signature(bbox),
                f"bbox={bbox}",
            )
            check(
                "Printed text à gauche exclu de la bbox",
                bbox.get("x", 0) > 150,
                f"bbox={bbox}",
            )
            check(
                "Date en haut à droite exclue de la bbox",
                bbox.get("y", 0) > 40,
                f"bbox={bbox}",
            )
            check(
                "extraction_quality dans [0,1]",
                0.0 <= body.get("extraction_quality", -1) <= 1.0,
                f"q={body.get('extraction_quality')}",
            )
            try:
                crop = decode_png_b64(body.get("signature_image_base64", ""))
                check(
                    "Crop PNG décodable",
                    crop is not None and crop.shape[0] > 0 and crop.shape[1] > 0,
                    f"crop={crop.shape if crop is not None else None}",
                )
            except Exception as exc:
                check("Crop PNG décodable", False, f"exception: {exc}")

            # Determinism: same image twice -> identical bbox
            status2, body2 = post_multipart(
                f"{dev_url}/api/signatures/extract",
                "file", "full.png", "image/png", png1,
            )
            check(
                "Déterminisme (bbox identique)",
                status2 == 200 and body2.get("signature_bbox") == bbox,
                f"bbox={bbox} bbox2={body2.get('signature_bbox')}",
            )

        # Debug: groups, selected index, reasons, images
        status, body = post_multipart(
            f"{dev_url}/api/signatures/debug", "file", "full.png", "image/png", png1
        )
        check("Debug full -> 200", status == 200, f"(HTTP {status})")
        if status == 200:
            check(
                "group_count >= 2 (plusieurs clusters)",
                body.get("group_count", 0) >= 2,
                f"groups={body.get('group_count')}",
            )
            groups = body.get("groups", [])
            check(
                "groups info présente",
                isinstance(groups, list) and len(groups) == body.get("group_count"),
                f"n={len(groups)}",
            )
            selected = body.get("selected_group_index", -2)
            check(
                "selected_group_index >= 0",
                selected >= 0,
                f"selected={selected}",
            )
            check(
                "Le groupe sélectionné est marqué selected",
                any(g.get("selected") for g in groups),
                f"groups={[(g.get('index'), g.get('selected')) for g in groups]}",
            )
            check(
                "selection_reason non vide",
                bool(body.get("selection_reason")),
                f"reason={body.get('selection_reason')[:60]}",
            )
            reasons = body.get("rejected_component_reasons", {})
            check(
                "micr_texte_imprime dans les raisons de rejet",
                reasons.get("micr_texte_imprime", 0) >= 1,
                f"reasons={reasons}",
            )
            for field in ("groups_image_base64", "group_image_base64"):
                try:
                    image = decode_png_b64(body.get(field, ""))
                    check(
                        f"{field} décodable",
                        image is not None and image.shape[0] > 0,
                        f"shape={image.shape if image is not None else None}",
                    )
                except Exception as exc:
                    check(f"{field} décodable", False, f"exception: {exc}")

        # ---- 2. Printed text near signature only -> signature still selected
        img2 = base_cheque()
        add_printed_text_left(img2)
        add_signature(img2)
        status, body = post_multipart(
            f"{dev_url}/api/signatures/extract", "file", "near_text.png", "image/png",
            encode_png(img2),
        )
        check("Texte imprimé proche -> 200", status == 200, f"(HTTP {status})")
        if status == 200:
            bbox = body.get("signature_bbox", {})
            check(
                "Signature sélectionnée malgré le texte proche",
                bbox_covers_signature(bbox),
                f"bbox={bbox}",
            )

        # ---- 3. Signature split into several nearby components -> one group
        img3 = base_cheque()
        cv2.ellipse(img3, (560, 270), (45, 18), 0, 0, 360, (20, 20, 20), -1)
        cv2.ellipse(img3, (650, 285), (40, 15), 0, 0, 360, (20, 20, 20), -1)
        cv2.line(img3, (610, 260), (630, 310), (20, 20, 20), 3)
        status, body = post_multipart(
            f"{dev_url}/api/signatures/extract", "file", "split.png", "image/png",
            encode_png(img3),
        )
        check("Signature divisée en composantes proches -> 200",
              status == 200, f"(HTTP {status})")
        if status == 200:
            bbox = body.get("signature_bbox", {})
            check(
                "bbox union couvre les composantes (large)",
                bbox.get("width", 0) > 150,
                f"bbox={bbox}",
            )
        status, body = post_multipart(
            f"{dev_url}/api/signatures/debug", "file", "split.png", "image/png",
            encode_png(img3),
        )
        if status == 200:
            check(
                "Un seul groupe pour composantes proches",
                body.get("group_count", -1) == 1,
                f"groups={body.get('group_count')}",
            )

        # ---- 4. Distant noise (specks + far blob) -> excluded from bbox
        img4 = base_cheque()
        add_signature(img4)
        rng = np.random.RandomState(42)
        for _ in range(40):
            x = rng.randint(ROI_X0, ROI_X0 + ROI_W - 3)
            y = rng.randint(ROI_Y0, ROI_Y0 + ROI_H - 3)
            rect(img4, x, y, x + 2, y + 2, (40, 40, 40))
        cv2.ellipse(img4, (400, 160), (25, 10), 0, 0, 360, (30, 30, 30), -1)
        status, body = post_multipart(
            f"{dev_url}/api/signatures/extract", "file", "noise.png", "image/png",
            encode_png(img4),
        )
        check("Bruit distant -> 200", status == 200, f"(HTTP {status})")
        if status == 200:
            bbox = body.get("signature_bbox", {})
            check(
                "bbox contient la signature",
                bbox_covers_signature(bbox),
                f"bbox={bbox}",
            )
            check(
                "Noise en haut à gauche exclu",
                bbox.get("y", 0) > 50,
                f"bbox={bbox}",
            )

        # ---- 5. Isolated '(28)'-like component -> excluded from bbox
        img5 = base_cheque()
        add_signature(img5)
        add_parentheses_28(img5)
        status, body = post_multipart(
            f"{dev_url}/api/signatures/extract", "file", "p28.png", "image/png",
            encode_png(img5),
        )
        check("'(28)' isolé -> 200", status == 200, f"(HTTP {status})")
        if status == 200:
            bbox = body.get("signature_bbox", {})
            check(
                "bbox contient la signature",
                bbox_covers_signature(bbox),
                f"bbox={bbox}",
            )
            check(
                "Composant '(28)' exclu (droite)",
                bbox.get("x", 0) + bbox.get("width", 0) < 379,
                f"bbox={bbox}",
            )

        # ---- 6. Date handwriting away from signature -> excluded
        img6 = base_cheque()
        add_signature(img6)
        add_date_upper_right(img6)
        status, body = post_multipart(
            f"{dev_url}/api/signatures/extract", "file", "date.png", "image/png",
            encode_png(img6),
        )
        check("Date éloignée -> 200", status == 200, f"(HTTP {status})")
        if status == 200:
            bbox = body.get("signature_bbox", {})
            check(
                "bbox contient la signature",
                bbox_covers_signature(bbox),
                f"bbox={bbox}",
            )
            check(
                "Date en haut exclue de la bbox",
                bbox.get("y", 0) > 40,
                f"bbox={bbox}",
            )

        # ---- 7. Blank image -> 422
        status, body = post_multipart(
            f"{dev_url}/api/signatures/extract", "file", "blank.png", "image/png",
            encode_png(base_cheque()),
        )
        check("Fond blanc sans signature -> 422", status == 422, f"(HTTP {status})")

        # ---- 8. MICR only -> 422 (rejected, nothing kept)
        img8 = base_cheque()
        add_micr_row(img8)
        status, body = post_multipart(
            f"{dev_url}/api/signatures/extract", "file", "micr.png", "image/png",
            encode_png(img8),
        )
        check("MICR seul -> 422", status == 422, f"(HTTP {status})")
        status, body = post_multipart(
            f"{dev_url}/api/signatures/debug", "file", "micr.png", "image/png",
            encode_png(img8),
        )
        if status == 200:
            check(
                "MICR seul -> retained == 0",
                body.get("retained_component_count", -1) == 0,
                f"retained={body.get('retained_component_count')}",
            )

    finally:
        dev_proc.terminate()
        try:
            dev_proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            dev_proc.kill()

    print(f"\n{'-' * 60}\nRésultats : {PASSED} passé(s), {FAILED} échoué(s)")
    return 0 if FAILED == 0 else 1


if __name__ == "__main__":
    sys.exit(main())

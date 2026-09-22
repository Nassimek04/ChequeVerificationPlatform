"""Integration tests for the V2.1 extraction pipeline.

Run with the venv activated:
    python tests/test_signature_extraction_v21.py

Starts uvicorn on ephemeral ports (development), runs the checks, then stops
the server. Covers the V2.1 behaviors (no rigid bottom crop):
- a long diagonal signature stroke descending into the bottom band is kept ;
- the final bbox includes the stroke extremities and stays bounded ;
- horizontal MICR / printed text in the bottom band is rejected ;
- signature + MICR in the same ROI -> signature kept, MICR reduced ;
- multiple close components are grouped into one signature group ;
- distant parasitic components are rejected from the bbox ;
- blank image -> reject ;
- crop PNG decodable ;
- extraction_quality always in [0, 1] ;
- /debug exposes retained/rejected component counts and images.
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


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def post_multipart(url: str, field: str, filename: str, content_type: str, data: bytes):
    boundary = "----v21boundary123"
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


def base_cheque():
    """Realistic cheque-like 800x355 image (ROI x 320..784, y 142..348)."""
    return make_image((355, 800))


def roi_of(image) -> dict:
    h, w = image.shape[:2]
    return {
        "x": round(0.40 * w),
        "y": round(0.40 * h),
        "width": round(0.98 * w) - round(0.40 * w),
        "height": round(0.98 * h) - round(0.40 * h),
    }


def add_signature(image, cx=600, cy=250, rx=45, ry=15):
    """A compact synthetic handwritten-like signature: ellipse + 2 strokes."""
    cv2.ellipse(image, (cx, cy), (rx, ry), 15, 0, 360, (20, 20, 20), -1)
    cv2.line(image, (cx - rx, cy - ry // 2), (cx - rx // 2, cy + ry), (20, 20, 20), 3)
    cv2.line(image, (cx + rx // 3, cy - ry), (cx + rx, cy + ry // 2), (20, 20, 20), 3)
    return image


def add_micr_row(image, roi):
    """A row of printed 'characters' inside the bottom MICR risk zone."""
    y0 = roi["y"] + round(roi["height"] * 0.91)
    for x in range(roi["x"] + 20, roi["x"] + 320, 30):
        cv2.rectangle(image, (x, y0), (x + 22, y0 + 8), (15, 15, 15), -1)
    return image


def diagonal_signature_cheque():
    """A flourish + a long diagonal stroke descending into the bottom band."""
    img = base_cheque()
    cv2.ellipse(img, (560, 235), (55, 12), 5, 0, 360, (20, 20, 20), -1)
    cv2.line(img, (430, 225), (770, 330), (20, 20, 20), 3)
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


def main() -> int:
    dev_port = free_port()
    dev_proc = start_server(dev_port, {"APP_ENV": "development"})
    dev_url = f"http://127.0.0.1:{dev_port}"

    try:
        if not wait_ready(dev_url, dev_proc):
            print("FAIL dev server did not start")
            return 1

        img = base_cheque()
        roi = roi_of(img)
        band_top = round(roi["height"] * (1 - 0.10))

        # ---- 1. Long trait diagonal descendant dans la bande basse -> conservé
        img_a = diagonal_signature_cheque()
        status, body = post_multipart(
            f"{dev_url}/api/signatures/extract", "file", "diagonal.png", "image/png",
            encode_png(img_a),
        )
        check("Trait diagonal -> 200", status == 200, f"(HTTP {status})")
        if status == 200:
            bbox = body.get("signature_bbox", {})
            check(
                "bbox descend dans la bande (trait conservé)",
                bbox.get("y", 0) + bbox.get("height", 0) >= band_top - 5,
                f"bbox={bbox} band_top={band_top}",
            )
            check(
                "bbox bornée dans la ROI",
                bbox.get("x", 0) >= 0 and bbox.get("y", 0) >= 0
                and bbox.get("width", 0) > 0 and bbox.get("height", 0) > 0
                and bbox.get("x", 0) + bbox.get("width", 0) <= roi["width"]
                and bbox.get("y", 0) + bbox.get("height", 0) <= roi["height"],
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

        # Debug: compteurs et images retained/rejected disponibles
        status, body = post_multipart(
            f"{dev_url}/api/signatures/debug", "file", "diagonal.png", "image/png",
            encode_png(img_a),
        )
        check("Debug diagonal -> 200", status == 200, f"(HTTP {status})")
        if status == 200:
            check(
                "total_component_count >= 1",
                isinstance(body.get("total_component_count"), int)
                and body.get("total_component_count", 0) >= 1,
                f"total={body.get('total_component_count')}",
            )
            check(
                "retained >= 1",
                isinstance(body.get("retained_component_count"), int)
                and body.get("retained_component_count", 0) >= 1,
                f"retained={body.get('retained_component_count')}",
            )
            for field in ("components_all_base64", "components_rejected_base64",
                          "components_image_base64", "group_image_base64"):
                try:
                    image = decode_png_b64(body.get(field, ""))
                    check(
                        f"{field} décodable",
                        image is not None and image.shape[0] > 0 and image.shape[1] > 0,
                        f"shape={image.shape if image is not None else None}",
                    )
                except Exception as exc:
                    check(f"{field} décodable", False, f"exception: {exc}")

        # ---- 2. Texte MICR horizontal en bas -> rejeté (aucun contenu)
        img_b = base_cheque()
        add_micr_row(img_b, roi)
        status, body = post_multipart(
            f"{dev_url}/api/signatures/extract", "file", "micr.png", "image/png",
            encode_png(img_b),
        )
        check("MICR seul -> rejet (422)", status == 422, f"(HTTP {status})")
        status, body = post_multipart(
            f"{dev_url}/api/signatures/debug", "file", "micr.png", "image/png",
            encode_png(img_b),
        )
        if status == 200:
            check(
                "MICR rejeté -> retained == 0",
                body.get("retained_component_count", -1) == 0,
                f"retained={body.get('retained_component_count')}",
            )
            check(
                "MICR rejeté -> micr_rejected_count >= 1",
                body.get("micr_rejected_count", 0) >= 1,
                f"micr_rejected={body.get('micr_rejected_count')}",
            )
            check(
                "MICR rejeté -> signature_bbox None",
                body.get("signature_bbox") is None,
            )

        # ---- 3. Signature + MICR dans la même ROI -> signature conservée
        img_c = base_cheque()
        add_signature(img_c, cx=600, cy=240, rx=70, ry=18)
        add_micr_row(img_c, roi)
        status, body = post_multipart(
            f"{dev_url}/api/signatures/extract", "file", "sig_micr.png", "image/png",
            encode_png(img_c),
        )
        check("Signature + MICR -> 200", status == 200, f"(HTTP {status})")
        if status == 200:
            bbox = body.get("signature_bbox", {})
            check(
                "MICR exclu de la bbox",
                bbox.get("y", 0) + bbox.get("height", 0) <= band_top,
                f"bbox={bbox} band_top={band_top}",
            )
            check(
                "extraction_quality dans [0,1]",
                0.0 <= body.get("extraction_quality", -1) <= 1.0,
                f"q={body.get('extraction_quality')}",
            )
        status, body = post_multipart(
            f"{dev_url}/api/signatures/debug", "file", "sig_micr.png", "image/png",
            encode_png(img_c),
        )
        if status == 200:
            check(
                "Signature + MICR -> retained >= 1",
                body.get("retained_component_count", 0) >= 1,
                f"retained={body.get('retained_component_count')}",
            )
            check(
                "Signature + MICR -> micr_rejected >= 1",
                body.get("micr_rejected_count", 0) >= 1,
                f"micr_rejected={body.get('micr_rejected_count')}",
            )

        # ---- 4. Plusieurs composantes proches -> regroupées (un seul groupe)
        img_d = base_cheque()
        cv2.ellipse(img_d, (560, 270), (45, 18), 0, 0, 360, (20, 20, 20), -1)
        cv2.ellipse(img_d, (650, 285), (40, 15), 0, 0, 360, (20, 20, 20), -1)
        status, body = post_multipart(
            f"{dev_url}/api/signatures/extract", "file", "grouped.png", "image/png",
            encode_png(img_d),
        )
        check("Composantes proches -> 200", status == 200, f"(HTTP {status})")
        if status == 200:
            bbox = body.get("signature_bbox", {})
            check(
                "bbox union couvre les deux composantes (large)",
                bbox.get("width", 0) > 140,
                f"bbox={bbox}",
            )
            check(
                "bbox bornée dans la ROI",
                bbox.get("x", 0) + bbox.get("width", 0) <= roi["width"]
                and bbox.get("y", 0) + bbox.get("height", 0) <= roi["height"],
                f"bbox={bbox}",
            )
        status, body = post_multipart(
            f"{dev_url}/api/signatures/debug", "file", "grouped.png", "image/png",
            encode_png(img_d),
        )
        if status == 200:
            check(
                "Un seul groupe pour composantes proches",
                body.get("group_count", -1) == 1,
                f"groups={body.get('group_count')}",
            )
            check(
                "retained == 2",
                body.get("retained_component_count", -1) == 2,
                f"retained={body.get('retained_component_count')}",
            )

        # ---- 5. Composante parasite éloignée -> rejetée de la bbox
        img_e = base_cheque()
        add_signature(img_e, cx=580, cy=245, rx=45, ry=15)
        cv2.ellipse(img_e, (730, 210), (22, 11), 0, 0, 360, (30, 30, 30), -1)
        status, body = post_multipart(
            f"{dev_url}/api/signatures/extract", "file", "parasite.png", "image/png",
            encode_png(img_e),
        )
        check("Parasite éloigné -> 200", status == 200, f"(HTTP {status})")
        if status == 200:
            bbox = body.get("signature_bbox", {})
            check(
                "bbox n'inclut pas le parasite éloigné",
                bbox.get("x", 0) + bbox.get("width", 0) < 370,
                f"bbox={bbox}",
            )
        status, body = post_multipart(
            f"{dev_url}/api/signatures/debug", "file", "parasite.png", "image/png",
            encode_png(img_e),
        )
        if status == 200:
            check(
                "Parasite éloigné -> 2 groupes",
                body.get("group_count", -1) == 2,
                f"groups={body.get('group_count')}",
            )

        # ---- 6. Image blanche -> rejet
        status, body = post_multipart(
            f"{dev_url}/api/signatures/extract", "file", "blank.png", "image/png",
            encode_png(base_cheque()),
        )
        check("Fond blanc sans signature -> 422", status == 422, f"(HTTP {status})")

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

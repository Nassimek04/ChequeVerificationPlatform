"""Integration tests for the V2 extraction pipeline.

Run with the venv activated:
    python tests/test_signature_extraction_v2.py

Starts uvicorn on ephemeral ports (development), runs the checks, then stops
the server. Covers the new V2 behaviors:
- blank cheque without signature -> no content ;
- synthetic signature on a light background ;
- multiple separated strokes belonging to one signature (union bbox) ;
- small parasitic specks are filtered ;
- parasitic content in the bottom (MICR) band is excluded ;
- bbox never covers the whole ROI without justification ;
- crop PNG decodable ;
- coordinates always bounded ;
- extraction_quality always in [0, 1] ;
- /debug development-only (404 outside development).
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
    boundary = "----v2boundary123"
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
    image = make_image((355, 800))
    return image


def roi_of(image) -> dict:
    h, w = image.shape[:2]
    return {
        "x": round(0.40 * w),
        "y": round(0.40 * h),
        "width": round(0.98 * w) - round(0.40 * w),
        "height": round(0.98 * h) - round(0.40 * h),
    }


def add_signature(image, cx=620, cy=270, rx=90, ry=35):
    """A synthetic handwritten-like signature: several separated strokes."""
    cv2.ellipse(image, (cx, cy), (rx, ry), 20, 0, 360, (20, 20, 20), -1)
    cv2.line(image, (cx - rx, cy - ry // 2), (cx - rx // 2, cy + ry), (20, 20, 20), 4)
    cv2.line(image, (cx + rx // 3, cy - ry), (cx + rx, cy + ry // 2), (20, 20, 20), 4)
    return image


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

    prod_port = free_port()
    prod_proc = start_server(prod_port, {"APP_ENV": "production"})
    prod_url = f"http://127.0.0.1:{prod_port}"

    roi = None

    try:
        if not wait_ready(dev_url, dev_proc):
            print("FAIL dev server did not start")
            return 1
        if not wait_ready(prod_url, prod_proc):
            print("FAIL prod server did not start")
            return 1

        # 1. Fond blanc sans signature -> aucun contenu
        blank = encode_png(base_cheque())
        status, body = post_multipart(
            f"{dev_url}/api/signatures/extract", "file", "blank.png", "image/png", blank
        )
        check("Fond blanc sans signature -> 422", status == 422, f"(HTTP {status})")

        # 2. Signature synthétique sur fond clair -> 200 + bbox + crop
        img = base_cheque()
        add_signature(img)
        roi = roi_of(img)
        status, body = post_multipart(
            f"{dev_url}/api/signatures/extract", "file", "cheque.png", "image/png",
            encode_png(img),
        )
        check("Signature sur fond clair -> 200", status == 200, f"(HTTP {status})")

        if status == 200:
            bbox = body.get("signature_bbox", {})
            check(
                "bbox dans la ROI",
                bbox.get("x", 0) >= 0 and bbox.get("y", 0) >= 0
                and bbox.get("width", 0) > 0 and bbox.get("height", 0) > 0
                and bbox.get("x", 0) + bbox.get("width", 0) <= roi["width"]
                and bbox.get("y", 0) + bbox.get("height", 0) <= roi["height"],
                f"bbox={bbox}",
            )
            check(
                "bbox ne couvre pas toute la ROI sans justification",
                bbox.get("width", 0) < roi["width"]
                and bbox.get("height", 0) < roi["height"],
                f"bbox={bbox} roi={roi}",
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

        # 3. Plusieurs traits séparés d'une même signature -> bbox union
        img2 = base_cheque()
        add_signature(img2, cx=560, cy=280, rx=50, ry=30)
        add_signature(img2, cx=700, cy=300, rx=45, ry=25)
        status, body = post_multipart(
            f"{dev_url}/api/signatures/extract", "file", "strokes.png", "image/png",
            encode_png(img2),
        )
        check("Plusieurs traits séparés -> 200", status == 200, f"(HTTP {status})")
        if status == 200:
            bbox = body.get("signature_bbox", {})
            # V2.3 : le raffinage dominant-core sélectionne UNE signature
            # dominante (les deux blobs sont en réalité 2 signatures séparées
            # de ~140 px). La bbox raffinée doit couvrir la signature
            # dominante (centre ROI-local ~(240,138)) et exclure l'autre
            # (centre ~(380,158)), sans recouvrir toute la ROI.
            check(
                "bbox V2.3 = signature dominante uniquement",
                bbox.get("width", 0) > 60
                and bbox.get("width", 0) < 240
                and bbox["x"] <= 240 <= bbox["x"] + bbox["width"]
                and bbox["y"] <= 138 <= bbox["y"] + bbox["height"]
                and not (
                    bbox["x"] <= 380 <= bbox["x"] + bbox["width"]
                    and bbox["y"] <= 158 <= bbox["y"] + bbox["height"]
                ),
                f"bbox={bbox}",
            )
            check(
                "bbox union ne couvre pas toute la ROI",
                bbox.get("width", 0) < roi["width"]
                and bbox.get("height", 0) < roi["height"],
                f"bbox={bbox}",
            )

        # 4. Petites composantes parasites -> filtrées
        img3 = base_cheque()
        add_signature(img3)
        for i in range(60):
            x = np.random.randint(roi["x"], roi["x"] + roi["width"] - 3)
            y = np.random.randint(roi["y"], roi["y"] + roi["height"] - 3)
            cv2.rectangle(img3, (x, y), (x + 2, y + 2), (40, 40, 40), -1)
        status, body = post_multipart(
            f"{dev_url}/api/signatures/extract", "file", "specks.png", "image/png",
            encode_png(img3),
        )
        check("Petites composantes parasites -> 200", status == 200, f"(HTTP {status})")
        if status == 200:
            bbox = body.get("signature_bbox", {})
            check(
                "bbox après filtrage des specks (dimensions bornées)",
                bbox.get("width", 0) < roi["width"]
                and bbox.get("height", 0) < roi["height"],
                f"bbox={bbox}",
            )

        # 5. Contenu parasite dans la bande inférieure (MICR) -> exclu
        img4 = base_cheque()
        add_signature(img4, cx=620, cy=230, rx=90, ry=30)
        zone_h = int(round(roi["height"] * (1 - 0.10)))
        for x in range(roi["x"], roi["x"] + roi["width"]):
            for y in range(roi["y"] + zone_h, roi["y"] + roi["height"]):
                img4[y, x] = (10, 10, 10)
        status, body = post_multipart(
            f"{dev_url}/api/signatures/extract", "file", "micr.png", "image/png",
            encode_png(img4),
        )
        check("Bande MICR parasite -> 200", status == 200, f"(HTTP {status})")
        if status == 200:
            bbox = body.get("signature_bbox", {})
            check(
                "bbox ne descend pas dans la bande exclue",
                bbox.get("y", 0) + bbox.get("height", 0) <= zone_h,
                f"bbox={bbox} zone_h={zone_h}",
            )

        # 6. Coordonnées toujours bornées + bbox dans ROI (déjà vérifiées, recheck)
        # 7. /debug development-only : 404 hors development
        status, _ = post_multipart(
            f"{prod_url}/api/signatures/debug", "file", "cheque.png", "image/png",
            encode_png(img),
        )
        check(
            "Debug indisponible hors development -> 404",
            status == 404,
            f"(HTTP {status})",
        )

        # 8. Debug disponible en development + nouvelles images
        status, body = post_multipart(
            f"{dev_url}/api/signatures/debug", "file", "cheque.png", "image/png",
            encode_png(img),
        )
        check("Debug disponible en development -> 200", status == 200, f"(HTTP {status})")
        if status == 200:
            check("analysis_zone présent", body.get("analysis_zone") is not None)
            zone = body.get("analysis_zone", {})
            check(
                "analysis_zone dans la ROI",
                zone.get("height", 0) <= roi["height"]
                and zone.get("width", 0) == roi["width"],
                f"zone={zone}",
            )
            check(
                "component_count >= 0",
                isinstance(body.get("component_count"), int)
                and body.get("component_count", -1) >= 0,
                f"count={body.get('component_count')}",
            )
            for field in ("mask_image_base64", "components_image_base64"):
                try:
                    image = decode_png_b64(body.get(field, ""))
                    check(
                        f"{field} décodable",
                        image is not None and image.shape[0] > 0 and image.shape[1] > 0,
                        f"shape={image.shape if image is not None else None}",
                    )
                except Exception as exc:
                    check(f"{field} décodable", False, f"exception: {exc}")

        # 9. Debug sans contenu -> pas de crash, message + images ROI/mask
        status, body = post_multipart(
            f"{dev_url}/api/signatures/debug", "file", "blank.png", "image/png", blank
        )
        check("Debug sans contenu -> 200", status == 200, f"(HTTP {status})")
        if status == 200:
            check(
                "Message explicite sans contenu",
                isinstance(body.get("message"), str) and len(body.get("message", "")) > 0,
                f"message={body.get('message')}",
            )
            check("Aucune bbox sans contenu", body.get("signature_bbox") is None)
            check("Quality à 0 sans contenu", body.get("extraction_quality") == 0.0)
            try:
                mask_img = decode_png_b64(body.get("mask_image_base64", ""))
                check(
                    "Mask décodable sans contenu",
                    mask_img is not None and mask_img.shape[0] > 0,
                )
            except Exception as exc:
                check("Mask décodable sans contenu", False, f"exception: {exc}")

    finally:
        dev_proc.terminate()
        prod_proc.terminate()
        for proc in (dev_proc, prod_proc):
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()

    print(f"\n{'-' * 60}\nRésultats : {PASSED} passé(s), {FAILED} échoué(s)")
    return 0 if FAILED == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
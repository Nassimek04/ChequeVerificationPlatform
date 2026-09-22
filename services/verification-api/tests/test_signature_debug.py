"""Integration tests for POST /api/signatures/debug (development-only).

Run with the venv activated:
    python tests/test_signature_debug.py

Starts uvicorn on ephemeral ports (one in development, one outside
development), runs the checks, then stops them.
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


def synthetic_cheque_with_signature() -> bytes:
    """Synthetic 1600x800 cheque with an ink blob in the bottom-right ROI."""
    image = np.full((800, 1600, 3), 245, dtype=np.uint8)
    cv2.rectangle(image, (200, 150), (1200, 220), (0, 0, 0), 3)  # fake text line
    cv2.ellipse(image, (1250, 620), (180, 70), 15, 0, 360, (10, 10, 10), -1)
    cv2.line(image, (1000, 600), (1450, 700), (10, 10, 10), 12)
    ok, encoded = cv2.imencode(".png", image)
    return encoded.tobytes()


def synthetic_cheque_without_content() -> bytes:
    """Solid light image with no detectable ink in the ROI."""
    image = np.full((800, 1600, 3), 250, dtype=np.uint8)
    ok, encoded = cv2.imencode(".png", image)
    return encoded.tobytes()


def post_multipart(url: str, field: str, filename: str, content_type: str, data: bytes):
    boundary = "----debugboundary789"
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


def decode_png_b64(value: str):
    decoded = base64.b64decode(value)
    image = cv2.imdecode(np.frombuffer(decoded, np.uint8), cv2.IMREAD_COLOR)
    return image


def check(name: str, condition: bool, detail: str = ""):
    global PASSED, FAILED
    if condition:
        PASSED += 1
        print(f"PASS  {name} {detail}")
    else:
        FAILED += 1
        print(f"FAIL  {name} {detail}")


def main() -> int:
    # ---- Development server (debug route registered) ----
    dev_port = free_port()
    dev_proc = start_server(dev_port, {"APP_ENV": "development"})
    dev_url = f"http://127.0.0.1:{dev_port}"

    # ---- Non-development server (debug route NOT registered) ----
    prod_port = free_port()
    prod_proc = start_server(prod_port, {"APP_ENV": "production"})
    prod_url = f"http://127.0.0.1:{prod_port}"

    try:
        if not wait_ready(dev_url, dev_proc):
            print("FAIL dev server did not start")
            return 1
        if not wait_ready(prod_url, prod_proc):
            print("FAIL prod server did not start")
            return 1

        # --- Debug disponible en development ---
        status, body = post_multipart(
            f"{dev_url}/api/signatures/debug", "file", "cheque.png", "image/png",
            synthetic_cheque_with_signature(),
        )
        check("Debug disponible en development -> 200", status == 200, f"(HTTP {status})")

        if status == 200:
            check(
                "Réponse complète debug",
                body.get("success") is True
                and body.get("original_width") == 1600
                and body.get("original_height") == 800
                and body.get("image_format") == "png",
                f"w={body.get('original_width')} h={body.get('original_height')}",
            )
            roi = body.get("candidate_roi", {})
            bbox = body.get("signature_bbox") or {}
            check(
                "ROI bornée dans l'image",
                roi.get("x", 0) >= 0
                and roi.get("y", 0) >= 0
                and roi.get("width", 0) > 0
                and roi.get("height", 0) > 0
                and roi.get("x", 0) + roi.get("width", 0) <= 1600
                and roi.get("y", 0) + roi.get("height", 0) <= 800,
                f"ROI={roi}",
            )
            check(
                "BBox présente et dans la ROI",
                bbox.get("width", 0) > 0
                and bbox.get("height", 0) > 0
                and bbox.get("x", 0) + bbox.get("width", 0) <= roi.get("width", 0)
                and bbox.get("y", 0) + bbox.get("height", 0) <= roi.get("height", 0),
                f"bbox={bbox}",
            )
            check(
                "extraction_quality dans [0,1]",
                0.0 <= body.get("extraction_quality", -1) <= 1.0,
                f"q={body.get('extraction_quality')}",
            )

            # Images Base64 décodables
            try:
                annotated = decode_png_b64(body.get("original_with_roi_base64", ""))
                check(
                    "Image originale+ROI décodable et même taille",
                    annotated is not None and annotated.shape[0] == 800 and annotated.shape[1] == 1600,
                    f"shape={annotated.shape if annotated is not None else None}",
                )

                roi_img = decode_png_b64(body.get("roi_image_base64", ""))
                check(
                    "Image ROI décodable et taille = ROI",
                    roi_img is not None
                    and roi_img.shape[1] == roi.get("width", 0)
                    and roi_img.shape[0] == roi.get("height", 0),
                    f"shape={roi_img.shape if roi_img is not None else None}",
                )

                sig_img = decode_png_b64(body.get("signature_image_base64", ""))
                check(
                    "Crop final décodable",
                    sig_img is not None and sig_img.shape[0] > 0 and sig_img.shape[1] > 0,
                    f"shape={sig_img.shape if sig_img is not None else None}",
                )
            except Exception as exc:
                check("Images Base64 décodables", False, f"exception: {exc}")

            # ROI dessinée correctement : l'image annotée diffère de l'originale
            source = cv2.imdecode(
                np.frombuffer(synthetic_cheque_with_signature(), np.uint8),
                cv2.IMREAD_COLOR,
            )
            annotated = decode_png_b64(body.get("original_with_roi_base64", ""))
            difference = int(np.count_nonzero(cv2.absdiff(annotated, source)))
            check(
                "ROI dessinée sur une copie (pixels modifiés)",
                annotated is not None and difference > 0,
                f"diff={difference}",
            )

        # --- Debug indisponible hors development ---
        status, _ = post_multipart(
            f"{prod_url}/api/signatures/debug", "file", "cheque.png", "image/png",
            synthetic_cheque_with_signature(),
        )
        check(
            "Debug indisponible hors development -> 404",
            status == 404,
            f"(HTTP {status})",
        )

        # --- Cas sans contenu : pas de crash, réponse partielle exploitable ---
        status, body = post_multipart(
            f"{dev_url}/api/signatures/debug", "file", "blank.png", "image/png",
            synthetic_cheque_without_content(),
        )
        check(
            "Cas sans contenu -> 200 (pas de crash)",
            status == 200,
            f"(HTTP {status})",
        )
        if status == 200:
            check(
                "Message explicite présent",
                isinstance(body.get("message"), str) and len(body.get("message", "")) > 0,
                f"message={body.get('message')}",
            )
            check(
                "Aucune bbox sans contenu",
                body.get("signature_bbox") is None,
            )
            check(
                "Aucun crop final sans contenu",
                body.get("signature_image_base64") is None,
            )
            check(
                "Quality à 0 sans contenu",
                body.get("extraction_quality") == 0.0,
                f"q={body.get('extraction_quality')}",
            )
            try:
                annotated = decode_png_b64(body.get("original_with_roi_base64", ""))
                check(
                    "Image originale+ROI décodable sans contenu",
                    annotated is not None and annotated.shape[0] == 800 and annotated.shape[1] == 1600,
                    f"shape={annotated.shape if annotated is not None else None}",
                )
                roi_img = decode_png_b64(body.get("roi_image_base64", ""))
                roi = body.get("candidate_roi", {})
                check(
                    "Image ROI décodable sans contenu",
                    roi_img is not None
                    and roi_img.shape[1] == roi.get("width", 0)
                    and roi_img.shape[0] == roi.get("height", 0),
                    f"shape={roi_img.shape if roi_img is not None else None}",
                )
                source = cv2.imdecode(
                    np.frombuffer(synthetic_cheque_without_content(), np.uint8),
                    cv2.IMREAD_COLOR,
                )
                difference = int(np.count_nonzero(cv2.absdiff(annotated, source)))
                check(
                    "ROI dessinée même sans contenu",
                    annotated is not None and difference > 0,
                    f"diff={difference}",
                )
            except Exception as exc:
                check("Images Base64 sans contenu", False, f"exception: {exc}")

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
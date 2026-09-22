"""Integration tests for POST /api/signatures/extract.

Run with the venv activated:
    python tests/test_signature_extraction.py

Starts uvicorn on an ephemeral port, runs the checks, then stops it.
"""

import base64
import os
import socket
import subprocess
import sys
import time
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
    # "signature" in the candidate ROI (x: 0.50..0.98 -> 800..1568, y: 0.55..0.95 -> 440..760)
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
    boundary = "----testboundary456"
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


def check(name: str, condition: bool, detail: str = ""):
    global PASSED, FAILED
    if condition:
        PASSED += 1
        print(f"PASS  {name} {detail}")
    else:
        FAILED += 1
        print(f"FAIL  {name} {detail}")


def main() -> int:
    port = free_port()
    process = subprocess.Popen(
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
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    url = f"http://127.0.0.1:{port}"
    try:
        for _ in range(30):
            try:
                urllib.request.urlopen(f"{url}/api/health", timeout=2)
                break
            except Exception:
                time.sleep(0.5)
        else:
            print("FAIL server did not start")
            process.terminate()
            return 1

        # Image valide avec contenu -> 200 + bbox + crop PNG décodable
        status, body = post_multipart(
            f"{url}/api/signatures/extract", "file", "cheque.png", "image/png",
            synthetic_cheque_with_signature(),
        )
        check("Image valide -> 200", status == 200, f"(HTTP {status})")
        if status == 200:
            check(
                "Réponse complète",
                body.get("success") is True
                and body.get("original_width") == 1600
                and body.get("original_height") == 800
                and body.get("image_format") == "png",
                f"w={body.get('original_width')} h={body.get('original_height')}",
            )
            roi = body.get("candidate_roi", {})
            bbox = body.get("signature_bbox", {})
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
                "BBox bornée dans la ROI",
                bbox.get("width", 0) > 0 and bbox.get("height", 0) > 0,
                f"bbox={bbox}",
            )
            check(
                "extraction_quality dans [0,1]",
                0.0 <= body.get("extraction_quality", -1) <= 1.0,
                f"q={body.get('extraction_quality')}",
            )
            try:
                decoded = base64.b64decode(body.get("signature_image_base64", ""))
                image = cv2.imdecode(np.frombuffer(decoded, np.uint8), cv2.IMREAD_COLOR)
                check(
                    "Crop PNG décodable",
                    image is not None and image.shape[0] > 0 and image.shape[1] > 0,
                    f"crop={image.shape if image is not None else None}",
                )
            except Exception as exc:
                check("Crop PNG décodable", False, f"exception: {exc}")

        # Image sans contenu -> rejet (aucun contenu détecté)
        status, body = post_multipart(
            f"{url}/api/signatures/extract", "file", "blank.png", "image/png",
            synthetic_cheque_without_content(),
        )
        check("Image sans contenu -> rejet", status == 422, f"(HTTP {status})")

        # Image invalide -> 400
        status, body = post_multipart(
            f"{url}/api/signatures/extract", "file", "fake.png", "image/png",
            b"not an image at all",
        )
        check("Image invalide -> 400", status == 400, f"(HTTP {status})")

        # Type non supporté -> 415
        status, body = post_multipart(
            f"{url}/api/signatures/extract", "file", "doc.txt", "text/plain", b"hello",
        )
        check("Type non supporté -> 415", status == 415, f"(HTTP {status})")

        # Fichier vide -> 400
        status, body = post_multipart(
            f"{url}/api/signatures/extract", "file", "empty.png", "image/png", b"",
        )
        check("Fichier vide -> 400", status == 400, f"(HTTP {status})")

    finally:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()

    print(f"\n{'-' * 60}\nRésultats : {PASSED} passé(s), {FAILED} échoué(s)")
    return 0 if FAILED == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
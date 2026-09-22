"""Integration tests for POST /api/images/analyze.

Run with the venv activated:
    python tests/test_image_analyze.py

Starts uvicorn on an ephemeral port, runs the checks, then stops it.
"""

import io
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
BASE_URL = "http://127.0.0.1:18080"

PASSED = 0
FAILED = 0


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def png_bytes() -> bytes:
    image = np.zeros((300, 400, 3), dtype=np.uint8)
    image[:] = (200, 200, 200)
    success, encoded = cv2.imencode(".png", image)
    if not success:
        raise RuntimeError("png encode failed")
    return encoded.tobytes()


def jpeg_bytes() -> bytes:
    image = np.zeros((200, 500, 3), dtype=np.uint8)
    image[:] = (120, 120, 120)
    success, encoded = cv2.imencode(".jpg", image)
    if not success:
        raise RuntimeError("jpeg encode failed")
    return encoded.tobytes()


def text_bytes() -> bytes:
    return b"this is definitely not an image"


def post_multipart(url: str, field: str, filename: str, content_type: str, data: bytes):
    boundary = "----testboundary123"
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


def check(name: str, expected: int, actual: int, body: dict | None = None, extra: str = ""):
    global PASSED, FAILED
    ok = actual == expected
    if ok:
        PASSED += 1
        print(f"PASS  {name}: HTTP {actual} (expected {expected}) {extra}")
    else:
        FAILED += 1
        print(f"FAIL  {name}: HTTP {actual} (expected {expected}) body={body}")


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

        # PNG valide -> 200
        status, body = post_multipart(
            f"{url}/api/images/analyze", "file", "cheque.png", "image/png", png_bytes()
        )
        check("PNG valide -> 200", 200, status, body,
              extra=f"w={body.get('width')} h={body.get('height')}" if status == 200 else "")

        # JPEG valide -> 200
        status, body = post_multipart(
            f"{url}/api/images/analyze", "file", "cheque.jpg", "image/jpeg", jpeg_bytes()
        )
        check("JPEG valide -> 200", 200, status, body,
              extra=f"w={body.get('width')} h={body.get('height')}" if status == 200 else "")

        # Texte envoyé comme image -> 400 (decode impossible)
        status, body = post_multipart(
            f"{url}/api/images/analyze", "file", "fake.png", "image/png", text_bytes()
        )
        check("Texte comme image -> 400", 400, status, body)

        # Fichier vide -> 400
        status, body = post_multipart(
            f"{url}/api/images/analyze", "file", "empty.png", "image/png", b""
        )
        check("Fichier vide -> 400", 400, status, body)

        # Type non supporté -> 415
        status, body = post_multipart(
            f"{url}/api/images/analyze", "file", "doc.txt", "text/plain", b"hello"
        )
        check("Type non supporté -> 415", 415, status, body)

    finally:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()

    print(f"\n{'-' * 50}\nRésultats : {PASSED} passé(s), {FAILED} échoué(s)")
    return 0 if FAILED == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
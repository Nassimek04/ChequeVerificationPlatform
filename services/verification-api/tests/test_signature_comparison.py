"""Integration tests for the experimental signature comparison V1.

Run with the venv activated:
    python tests/test_signature_comparison.py

Starts uvicorn on an ephemeral port (development), runs the checks, then stops
the server. Covers the V1 comparison behaviors:
- identical signatures -> very high score ;
- same signature resized -> high score ;
- same signature with different padding -> high score ;
- same signature with a small translation -> high score ;
- clearly different signatures -> score lower than the identical case ;
- blank (white) image -> rejected ;
- undecodable file -> 400 ;
- unsupported MIME type -> 415 ;
- score always in [0, 1] ;
- deterministic (same input -> same score).
"""

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


def post_multipart(url: str, fields: dict):
    boundary = "----compareboundary123"
    parts = []
    for field, (filename, content_type, data) in fields.items():
        parts.append(
            f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="{field}"; filename="{filename}"\r\n'
            f"Content-Type: {content_type}\r\n\r\n"
        )
        parts.append(data)
        parts.append("\r\n")
    parts.append(f"--{boundary}--\r\n")

    body = b"".join(p.encode() if isinstance(p, str) else p for p in parts)
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


def make_signature(variant=0, w=300, h=150):
    """A synthetic signature-like image (dark ink on a light background)."""
    cx, cy = w // 2, h // 2
    img = np.full((h, w, 3), 245, dtype=np.uint8)
    if variant == 0:
        cv2.ellipse(img, (cx, cy), (90, 40), 20, 0, 360, (20, 20, 20), -1)
        cv2.line(img, (cx - 90, cy - 20), (cx - 45, cy + 40), (20, 20, 20), 4)
        cv2.line(img, (cx + 30, cy - 40), (cx + 90, cy + 20), (20, 20, 20), 4)
    else:
        cv2.circle(img, (cx, cy), 55, (20, 20, 20), -1)
        cv2.line(img, (0, cy + 40), (w, cy + 40), (20, 20, 20), 6)
    return img


def encode_png(image: np.ndarray) -> bytes:
    ok, encoded = cv2.imencode(".png", image)
    return encoded.tobytes()


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
    env.update({"APP_ENV": "development"})
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


def compare(url: str, extracted: bytes, reference: bytes, mime_a: str = "image/png", mime_b: str = "image/png"):
    return post_multipart(
        f"{url}/api/signatures/compare",
        {
            "extracted_file": ("extracted.png", mime_a, extracted),
            "reference_file": ("reference.png", mime_b, reference),
        },
    )


def main() -> int:
    port = free_port()
    proc = start_server(port)
    url = f"http://127.0.0.1:{port}"

    sig = make_signature(0)
    sig_big = cv2.resize(sig, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    sig_small = cv2.resize(sig, None, fx=0.75, fy=0.75, interpolation=cv2.INTER_CUBIC)

    gray = cv2.cvtColor(sig, cv2.COLOR_BGR2GRAY)
    ys, xs = np.where(gray < 200)
    ink_tight = sig[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    ink_padded = np.full((250, 400, 3), 245, dtype=np.uint8)
    ink_padded[50:50 + ink_tight.shape[0], 60:60 + ink_tight.shape[1]] = ink_tight

    shift = np.float32([[1, 0, 4], [0, 1, 3]])
    sig_shifted = cv2.warpAffine(sig, shift, (300, 150), borderValue=(245, 245, 245))

    sig_diff = make_signature(1)
    blank = np.full((300, 150, 3), 255, dtype=np.uint8)

    try:
        if not wait_ready(url, proc):
            print("FAIL server did not start")
            return 1

        # 1. Identical -> very high score
        status, body = compare(url, encode_png(sig), encode_png(sig))
        check("Identiques -> 200", status == 200, f"(HTTP {status})")
        check("Score identique très élevé", body.get("similarity_score", -1) >= 0.95,
              f"score={body.get('similarity_score')}")

        # 2. Same signature resized -> high score
        status, body = compare(url, encode_png(sig), encode_png(sig_big))
        check("Resize x2 -> 200", status == 200, f"(HTTP {status})")
        check("Score resize élevé", body.get("similarity_score", -1) >= 0.9,
              f"score={body.get('similarity_score')}")

        # 3. Same signature, different padding -> high score
        status, body = compare(url, encode_png(ink_tight), encode_png(ink_padded))
        check("Padding différent -> 200", status == 200, f"(HTTP {status})")
        check("Score padding élevé", body.get("similarity_score", -1) >= 0.9,
              f"score={body.get('similarity_score')}")

        # 4. Small translation -> high score
        status, body = compare(url, encode_png(sig), encode_png(sig_shifted))
        check("Translation légère -> 200", status == 200, f"(HTTP {status})")
        check("Score translation élevé", body.get("similarity_score", -1) >= 0.9,
              f"score={body.get('similarity_score')}")

        # 5. Clearly different signatures -> lower than the identical case
        _, body_identical = compare(url, encode_png(sig), encode_png(sig))
        status, body_diff = compare(url, encode_png(sig), encode_png(sig_diff))
        check("Signatures différentes -> 200", status == 200, f"(HTTP {status})")
        check(
            "Score différentes < score identiques",
            body_diff.get("similarity_score", 1) < body_identical.get("similarity_score", 0),
            f"diff={body_diff.get('similarity_score')} ident={body_identical.get('similarity_score')}",
        )
        check(
            "Score différentes nettement inférieur",
            body_diff.get("similarity_score", 1) < 0.9,
            f"score={body_diff.get('similarity_score')}",
        )

        # 6. Blank image -> rejected
        status, body = compare(url, encode_png(blank), encode_png(sig))
        check("Image blanche -> rejet", status == 422, f"(HTTP {status})")
        check("Message d'erreur explicite", "signature" in body.get("detail", "").lower(),
              f"detail={body.get('detail')}")

        # 7. Undecodable file -> 400
        status, body = compare(url, b"not an image at all", encode_png(sig))
        check("Fichier invalide -> 400", status == 400, f"(HTTP {status})")

        # 8. Unsupported MIME type -> 415
        status, body = compare(url, encode_png(sig), encode_png(sig), mime_a="image/gif")
        check("MIME non supporté -> 415", status == 415, f"(HTTP {status})")

        # 9. Score always in [0, 1]
        ok_range = True
        for pair in [(sig, sig), (sig, sig_big), (sig, sig_diff), (sig_small, sig_big)]:
            status, body = compare(url, encode_png(pair[0]), encode_png(pair[1]))
            score = body.get("similarity_score", -1) if status == 200 else -1
            if not (0.0 <= score <= 1.0):
                ok_range = False
        check("Score toujours dans [0,1]", ok_range)

        # 10. Determinism
        _, first = compare(url, encode_png(sig), encode_png(sig_big))
        _, second = compare(url, encode_png(sig), encode_png(sig_big))
        check(
            "Déterministe",
            first.get("similarity_score") == second.get("similarity_score"),
            f"{first.get('similarity_score')} vs {second.get('similarity_score')}",
        )
        check(
            "Métriques exposées",
            "metrics" in first and first["metrics"].get("mask_overlap") is not None,
            f"metrics={first.get('metrics')}",
        )
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()

    print(f"\n{len('test_signature_comparison')} file: {PASSED} passed, {FAILED} failed")
    return 0 if FAILED == 0 else 1


if __name__ == "__main__":
    sys.exit(main())

"""Exercise the local ASGI API with a synthetic image and the configured checkpoint."""

from __future__ import annotations

import io
import json
import socket
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np
from PIL import Image

PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from api import app as api_module  # noqa: E402
from uvicorn import Config, Server  # noqa: E402


def main() -> None:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    server = Server(Config(api_module.app, host="127.0.0.1", port=port,
                           log_level="error", access_log=False))
    thread = threading.Thread(target=server.run, name="local-api-smoke", daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{port}"
    try:
        health = None
        for _ in range(50):
            try:
                with urllib.request.urlopen(f"{base}/health", timeout=2) as response:
                    health = json.loads(response.read())
                break
            except (urllib.error.URLError, TimeoutError):
                time.sleep(0.2)
        if health is None:
            raise RuntimeError("Local API did not respond to /health.")
        if not health.get("model_available"):
            raise RuntimeError("The configured checkpoint is not available to the API.")

        image_bytes = io.BytesIO()
        Image.fromarray(np.full((32, 32, 3), 127, dtype=np.uint8)).save(image_bytes, format="PNG")
        boundary = "NexviaSyntheticSmokeBoundary"
        encoded_boundary = boundary.encode("ascii")
        payload = (
            b"--" + encoded_boundary
            + b'\r\nContent-Disposition: form-data; name="file"; filename="synthetic-smoke.png"\r\n'
            + b"Content-Type: image/png\r\n\r\n"
            + image_bytes.getvalue()
            + b"\r\n--" + encoded_boundary + b"--\r\n"
        )
        request = urllib.request.Request(
            f"{base}/predict", data=payload,
            headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=90) as response:
            result = json.loads(response.read())
            status_code = response.status
        if status_code != 200 or len(result.get("findings", [])) != 14:
            raise RuntimeError("/predict did not return 14 findings with HTTP 200.")
        if not result.get("model_metadata", {}).get("checkpoint_sha256"):
            raise RuntimeError("/predict omitted checkpoint identity metadata.")
        if "not a diagnosis" not in result.get("disclaimer", ""):
            raise RuntimeError("/predict omitted the educational-use disclaimer.")
        print(json.dumps({
            "health_status": health.get("status"),
            "predict_http_status": status_code,
            "model": result.get("model"),
            "finding_count": len(result["findings"]),
            "checkpoint_sha256": result["model_metadata"]["checkpoint_sha256"],
            "score_semantics": result.get("score_semantics"),
            "synthetic_input_only": True,
        }, indent=2))
    finally:
        server.should_exit = True
        thread.join(timeout=10)
        if thread.is_alive():
            server.force_exit = True
            raise RuntimeError("The local ASGI smoke server did not shut down cleanly.")


if __name__ == "__main__":
    main()

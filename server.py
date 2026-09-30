#!/usr/bin/env python3
"""Local FERMI server: the interactive demos + a small JSON API.

    python server.py                    # http://127.0.0.1:8765  (GPU if available, otherwise CPU)
    python server.py --device cpu --checkpoint checkpoints/step-600 --port 9000

API
    POST /api/classify   {"state": ..., "questions": {id: {type, instructions, criteria}}}
                         -> {"answers": {id: ...}, "ms": 123}
    GET  /api/info       model, device, checkpoint, evaluation
"""
import argparse
import json
import os
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from fermi import Fermi
from fermi.model import DEFAULT_CKPT

HERE = os.path.dirname(os.path.abspath(__file__))
DEMOS = os.path.join(HERE, "demos")
TYPES = {".html": "text/html; charset=utf-8", ".js": "text/javascript; charset=utf-8", ".css": "text/css; charset=utf-8",
         ".json": "application/json", ".svg": "image/svg+xml", ".png": "image/png"}
MAX_BODY = 2_000_000


def make_handler(model, info):
    class H(BaseHTTPRequestHandler):
        server_version = "fermi"

        def log_message(self, fmt, *args):
            pass

        def send(self, code, body, ctype="application/json"):
            if isinstance(body, (dict, list)):
                body = json.dumps(body, ensure_ascii=False)
            if isinstance(body, str):
                body = body.encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Headers", "Content-Type")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_OPTIONS(self):
            self.send(204, b"")

        def do_GET(self):
            path = self.path.split("?")[0]
            if path == "/api/info":
                return self.send(200, info)
            if path == "/":
                path = "/index.html"
            f = os.path.normpath(os.path.join(DEMOS, path.lstrip("/")))
            if not f.startswith(DEMOS + os.sep) or not os.path.isfile(f):
                return self.send(404, {"error": "not found"})
            with open(f, "rb") as fh:
                self.send(200, fh.read(), TYPES.get(os.path.splitext(f)[1], "application/octet-stream"))

        def do_POST(self):
            if self.path.split("?")[0] != "/api/classify":
                return self.send(404, {"error": "not found"})
            n = int(self.headers.get("Content-Length") or 0)
            if n <= 0 or n > MAX_BODY:
                return self.send(413, {"error": "body missing or too large"})
            try:
                req = json.loads(self.rfile.read(n))
                t = time.time()
                ans = model.classify(req["state"], req["questions"])
                self.send(200, {"answers": ans, "ms": int(1000 * (time.time() - t))})
            except (KeyError, ValueError, TypeError) as e:
                self.send(400, {"error": str(e)})
    return H


def main():
    ap = argparse.ArgumentParser(description="FERMI demos and API")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--device", default="auto", help="auto, cuda, cpu, mps")
    ap.add_argument("--checkpoint", default=DEFAULT_CKPT)
    ap.add_argument("--base", default=None, help="Hugging Face id or local folder of Qwen3.5-0.8B")
    a = ap.parse_args()
    model = Fermi.load(a.checkpoint, base=a.base, device=a.device)
    info = {"model": "FERMI-0.8B", "device": str(model.device), "dtype": str(model.dtype).replace("torch.", ""),
            "checkpoint": os.path.basename(os.path.normpath(a.checkpoint)), "base": model.config["base_model"],
            "evaluation": model.config.get("evaluation")}
    s = ThreadingHTTPServer((a.host, a.port), make_handler(model, info))
    s.daemon_threads = True
    print("FERMI ready: http://%s:%d" % (a.host, a.port), flush=True)
    try:
        s.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()

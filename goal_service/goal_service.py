"""HTTP goal-analysis service, to run on the laptop / PC.

    python goal_service.py [port=8765]

  GET  /health   -> {"ok": true, "stream": {...}}
  POST /analyze  -> body = raw .h264 file; response = JSON from Analyzer.analyze (goal type, confidence...)

Live analysis (see stream_analyzer.py), so that the goal type is ready almost immediately after the goal:
  POST /fragment?seq=N     -> body = one .h264 fragment (10 frames) finished on the Pi; N = increasing counter
  POST /goal?last_seq=N    -> N = last fragment finished at the infrared signal; response = goal type (waits ~0.6 s more)

If there is no answer in time, foos-master carries on as before.
"""
import json
import sys
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from analyzer import Analyzer
from stream_analyzer import StreamAnalyzer

LOCK = threading.Lock()   # one job at a time on the GPU
ANALYZER = None
STREAM = None


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"      # persistent connection: one fragment every 0.2 s

    def _send(self, code, obj):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/health":
            st = dict(STREAM.stats) if STREAM else {}
            if STREAM:
                st["done_seq"] = STREAM.done_seq
                st["queued"] = len(STREAM.in_q)
                lg = sorted(STREAM.lags)
                if lg:
                    st["lag_p50"], st["lag_p95"], st["lag_max"] = lg[len(lg) // 2], lg[int(len(lg) * .95)], lg[-1]
                    parts = list(STREAM.lag_parts)
                    st["lag_decode_s"] = max(p[0] for p in parts)
                    st["lag_wait_gpu_s"] = max(p[1] for p in parts)
            self._send(200, {"ok": True, "stream": st})
        else:
            self._send(404, {"error": "not found"})

    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        data = self.rfile.read(n)
        q = parse_qs(urlparse(self.path).query)
        route = urlparse(self.path).path
        if route == "/fragment":
            STREAM.add_fragment(int(q["seq"][0]), data)
            return self._send(200, {"ok": True})
        if route == "/goal":
            try:
                res = STREAM.analyze_goal(int(q["last_seq"][0]), extra=int(q.get("extra", ["3"])[0]))
            except Exception as e:
                return self._send(500, {"type": "unknown", "error": str(e)})
            print("goal (live): %s" % res, flush=True)
            return self._send(200, res)
        if route != "/analyze":
            return self._send(404, {"error": "not found"})
        fps = float(q.get("fps", ["49"])[0])
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "clip.h264"
            p.write_bytes(data)
            try:
                with LOCK:
                    res = ANALYZER.analyze(str(p), fps=fps)
            except Exception as e:   # never let the game crash
                return self._send(500, {"type": "unknown", "error": str(e)})
        print("analyzed: %s" % res, flush=True)
        self._send(200, res)

    def log_message(self, *a):
        pass


def main(port=8765):
    global ANALYZER, STREAM
    ANALYZER = Analyzer()
    STREAM = StreamAnalyzer(ANALYZER, LOCK)
    print("ready on port %d" % port, flush=True)
    ThreadingHTTPServer(("0.0.0.0", port), Handler).serve_forever()


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 8765)

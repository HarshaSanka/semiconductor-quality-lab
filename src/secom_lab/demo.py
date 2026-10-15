"""Portable local dashboard. Standard-library HTTP server plus the core ML runtime."""
import argparse
import json
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

from .data import read_csv
from .inference import load_bundle, predict


def create_handler(directory):
    directory = Path(directory)
    model = load_bundle(directory)

    class Handler(BaseHTTPRequestHandler):
        def send(self, status, content, kind="application/json"):
            self.send_response(status)
            self.send_header("Content-Type", kind)
            self.send_header("Content-Length", str(len(content)))
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(content)

        def do_GET(self):
            routes = {"/api/results": "results.json", "/api/leaderboard": "leaderboard.csv",
                      "/api/missingness": "missingness.json", "/example.csv": "example_runs.csv"}
            if self.path == "/":
                self.send(200, (Path(__file__).parent / "web.html").read_bytes(), "text/html; charset=utf-8")
            elif self.path in routes:
                name = routes[self.path]
                self.send(200, (directory / name).read_bytes(), "text/csv" if name.endswith(".csv") else "application/json")
            else:
                self.send(404, b'{"error":"Not found"}')

        def do_POST(self):
            if self.path != "/api/predict":
                self.send(404, b'{"error":"Not found"}')
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= 10_000_000:
                    raise ValueError("Upload 1 byte to 10 MB")
                frame = read_csv(self.rfile.read(length))
                self.send(200, json.dumps(predict(model, frame), allow_nan=False).encode())
            except (ValueError, TypeError, UnicodeError) as error:
                self.send(422, json.dumps({"error": str(error)}).encode())

    return Handler


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifacts", type=Path, default=Path("artifacts"))
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    server = HTTPServer(("127.0.0.1", args.port), create_handler(args.artifacts))
    print(f"Open http://127.0.0.1:{args.port} · Ctrl+C to stop", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()

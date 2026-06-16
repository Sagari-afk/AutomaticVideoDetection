import argparse
import json
import mimetypes
from datetime import datetime
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse


class EvaluationRequestHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, run_dir=None, **kwargs):
        self.run_dir = Path(run_dir).resolve()
        self.results = self._load_results()
        self.media_by_id = {
            item["id"]: item["video_path"]
            for item in self.results.get("items", [])
            if item.get("id") and item.get("video_path")
        }
        super().__init__(*args, directory=str(self.run_dir), **kwargs)

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path.startswith("/media/"):
            item_id = unquote(parsed.path[len("/media/"):])
            return self._serve_media(item_id)
        return super().do_GET()

    def do_POST(self):
        parsed = urlparse(self.path)
        if parsed.path == "/api/save":
            return self._save_ratings()
        self.send_error(404, "Not found")

    def _serve_media(self, item_id):
        media_path = self.media_by_id.get(item_id)
        if not media_path:
            self.send_error(404, "Unknown media id")
            return

        path = Path(media_path)
        if not path.exists():
            self.send_error(404, "Media file not found")
            return

        content_type = mimetypes.guess_type(str(path))[0] or "application/octet-stream"
        file_size = path.stat().st_size
        range_header = self.headers.get("Range")

        if range_header:
            start, end = self._parse_range_header(range_header, file_size)
            self.send_response(206)
            self.send_header("Content-Type", content_type)
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Content-Range", f"bytes {start}-{end}/{file_size}")
            self.send_header("Content-Length", str(end - start + 1))
            self.end_headers()
            self._write_file_range(path, start, end)
            return

        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(file_size))
        self.end_headers()

        with path.open("rb") as handle:
            while True:
                chunk = handle.read(1024 * 1024)
                if not chunk:
                    break
                self.wfile.write(chunk)

    @staticmethod
    def _parse_range_header(range_header, file_size):
        range_value = range_header.replace("bytes=", "", 1)
        start_text, _, end_text = range_value.partition("-")
        start = int(start_text) if start_text else 0
        end = int(end_text) if end_text else file_size - 1
        start = max(0, min(start, file_size - 1))
        end = max(start, min(end, file_size - 1))
        return start, end

    def _write_file_range(self, path, start, end):
        remaining = end - start + 1
        with path.open("rb") as handle:
            handle.seek(start)
            while remaining > 0:
                chunk = handle.read(min(1024 * 1024, remaining))
                if not chunk:
                    break
                self.wfile.write(chunk)
                remaining -= len(chunk)

    def _save_ratings(self):
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length)

        try:
            payload = json.loads(raw.decode("utf-8"))
        except json.JSONDecodeError:
            self.send_error(400, "Invalid JSON")
            return

        payload["updated_at"] = payload.get("updated_at") or datetime.now().isoformat(timespec="seconds")
        ratings_path = self.run_dir / "ratings.json"
        ratings_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(b'{"ok": true}')

    def _load_results(self):
        results_path = self.run_dir / "results.json"
        if not results_path.exists():
            return {"items": []}
        return json.loads(results_path.read_text(encoding="utf-8"))


def parse_args():
    parser = argparse.ArgumentParser(description="Serve evaluation review UI and save ratings.")
    parser.add_argument("--run-dir", required=True, help="Path to EvaluationRuns/run_* directory")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    return parser.parse_args()


def main():
    args = parse_args()

    def handler(*handler_args, **handler_kwargs):
        return EvaluationRequestHandler(*handler_args, run_dir=args.run_dir, **handler_kwargs)

    server = ThreadingHTTPServer((args.host, args.port), handler)
    url = f"http://{args.host}:{args.port}/review.html"
    print(f"Serving evaluation UI: {url}")
    print("Press Ctrl+C to stop.")
    server.serve_forever()


if __name__ == "__main__":
    main()

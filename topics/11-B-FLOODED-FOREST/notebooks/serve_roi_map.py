#!/usr/bin/env python3
"""Serve the ROI world map (notebooks/roi_map.html) on a local web server.

The page reads ../rois.json live, so edits to rois.json show up on a browser refresh.
A local server is needed because browsers do not let a page opened from disk read local files.

    python notebooks/serve_roi_map.py            # http://localhost:8000/notebooks/roi_map.html
    python notebooks/serve_roi_map.py --port 8080 --no-browser
"""

import argparse
import functools
import http.server
import webbrowser
from pathlib import Path

TOPIC_DIR = Path(__file__).resolve().parents[1]   # serves the topic folder (rois.json + notebooks/)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--no-browser", action="store_true", help="do not open the browser")
    args = parser.parse_args()

    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(TOPIC_DIR))
    with http.server.ThreadingHTTPServer(("127.0.0.1", args.port), handler) as server:
        url = f"http://localhost:{args.port}/notebooks/roi_map.html"
        print(f"Serving {TOPIC_DIR}\nROI map: {url}\n(Ctrl+C to stop)")
        if not args.no_browser:
            webbrowser.open(url)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            print("\nstopped")


if __name__ == "__main__":
    main()

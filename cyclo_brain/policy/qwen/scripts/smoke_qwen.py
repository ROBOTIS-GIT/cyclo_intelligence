#!/usr/bin/env python3
"""Standard-library client: optionally load weights, then infer text/image."""

import argparse
import base64
import json
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:7110")
    parser.add_argument("--load", action="store_true")
    parser.add_argument("--image", type=Path)
    parser.add_argument("--prompt", default="한국어로 준비되었다고 한 문장으로 답해줘.")
    parser.add_argument("--thinking", action="store_true")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    args = parser.parse_args()

    def post(path, payload=None):
        data = json.dumps(payload or {}).encode()
        request = Request(args.url.rstrip("/") + path, data=data,
                          headers={"Content-Type": "application/json"})
        try:
            with urlopen(request, timeout=7200 if path == "/load" else 300) as response:
                return json.load(response)
        except HTTPError as exc:
            raise SystemExit(f"HTTP {exc.code}: {exc.read().decode()}") from exc

    if args.load:
        print(json.dumps(post("/load"), ensure_ascii=False))
    payload = {"prompt": args.prompt, "enable_thinking": args.thinking,
               "max_new_tokens": args.max_new_tokens}
    if args.image:
        payload["image_base64"] = base64.b64encode(args.image.read_bytes()).decode()
    result = post("/generate", payload)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not result["text"] or result["finish_reason"] != "stop":
        raise SystemExit("Incomplete answer: increase --max-new-tokens or disable --thinking")


if __name__ == "__main__":
    main()

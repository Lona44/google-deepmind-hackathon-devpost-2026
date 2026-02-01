#!/usr/bin/env python3
"""Test what happens when we hit Kimi's rate limit.

Makes rapid requests to see if Kimi returns 429 or just hangs.

Usage:
    python scripts/test_rate_limit.py
"""

import json
import os
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

from dotenv import load_dotenv

load_dotenv()


def make_request(request_num: int) -> dict:
    """Make a single request to Kimi and return timing/result."""
    api_key = os.environ.get("MOONSHOT_API_KEY")
    url = "https://api.moonshot.ai/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    payload = {
        "model": "kimi-k2.5",
        "messages": [{"role": "user", "content": f"Say {request_num}"}],
        "max_tokens": 10,
    }

    start = time.time()
    try:
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(url, data=data, headers=headers, method="POST")

        with urllib.request.urlopen(req, timeout=30) as response:
            elapsed = time.time() - start
            response.read()  # Consume response body
            return {
                "request": request_num,
                "status": response.status,
                "elapsed": elapsed,
                "success": True,
            }

    except urllib.error.HTTPError as e:
        elapsed = time.time() - start
        error_body = e.read().decode("utf-8")
        return {
            "request": request_num,
            "status": e.code,
            "elapsed": elapsed,
            "success": False,
            "error": error_body[:200],
        }

    except Exception as e:
        elapsed = time.time() - start
        return {
            "request": request_num,
            "status": None,
            "elapsed": elapsed,
            "success": False,
            "error": str(e)[:200],
        }


def main():
    print("Testing Kimi rate limit behavior...")
    print("Making 10 rapid concurrent requests to see what happens.\n")

    # Make 10 concurrent requests
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(make_request, i) for i in range(10)]

        for future in as_completed(futures):
            result = future.result()
            status_str = f"HTTP {result['status']}" if result["status"] else "ERROR"
            success_str = "✓" if result["success"] else "✗"
            elapsed_str = f"{result['elapsed']:.2f}s"

            print(f"  Request {result['request']:2d}: {success_str} {status_str} ({elapsed_str})")
            if not result["success"] and "error" in result:
                print(f"             Error: {result['error'][:100]}")

    print("\nIf you see HTTP 429, Kimi returns rate limit errors.")
    print("If all requests hang or timeout, Kimi queues instead of rejecting.")


if __name__ == "__main__":
    main()

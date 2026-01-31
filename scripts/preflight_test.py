#!/usr/bin/env python3
"""Standalone preflight test to check API connectivity and response.

Usage:
    python scripts/preflight_test.py          # Test Kimi (default)
    python scripts/preflight_test.py gemini   # Test Gemini
    python scripts/preflight_test.py claude   # Test Claude
"""

import json
import os
import sys
import urllib.error
import urllib.request

from dotenv import load_dotenv

load_dotenv()


def preflight_test(model: str = "kimi") -> None:
    """Test API connectivity and print full response details."""
    print(f"Testing {model} API...\n")

    if model == "kimi":
        api_key = os.environ.get("MOONSHOT_API_KEY")
        if not api_key:
            print("ERROR: MOONSHOT_API_KEY not set")
            sys.exit(1)

        url = "https://api.moonshot.ai/v1/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        }
        payload = {
            "model": "kimi-k2.5",
            "messages": [{"role": "user", "content": "Say OK"}],
            "max_tokens": 5,
        }

    elif model in ("gemini", "gemini2.5", "robotics"):
        api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
        if not api_key:
            print("ERROR: GOOGLE_API_KEY or GEMINI_API_KEY not set")
            sys.exit(1)

        model_name = "gemini-2.5-flash"
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={api_key}"
        headers = {"Content-Type": "application/json"}
        payload = {
            "contents": [{"parts": [{"text": "Say OK"}]}],
            "generationConfig": {"maxOutputTokens": 5},
        }

    elif model in ("claude", "opus"):
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            print("ERROR: ANTHROPIC_API_KEY not set")
            sys.exit(1)

        url = "https://api.anthropic.com/v1/messages"
        headers = {
            "Content-Type": "application/json",
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
        }
        payload = {
            "model": "claude-3-haiku-20240307",
            "max_tokens": 5,
            "messages": [{"role": "user", "content": "Say OK"}],
        }

    else:
        print(f"ERROR: Unknown model '{model}'")
        print("Supported: kimi, gemini, claude, opus")
        sys.exit(1)

    # Make the request
    print(f"URL: {url.split('?')[0]}...")  # Hide API key in URL
    print(f"Payload: {json.dumps(payload, indent=2)}")
    print()

    try:
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(url, data=data, headers=headers, method="POST")

        print("Sending request (timeout: 30s)...")
        with urllib.request.urlopen(req, timeout=30) as response:
            status = response.status
            response_headers = dict(response.headers)
            result = json.loads(response.read().decode("utf-8"))

            print(f"\n=== SUCCESS ===")
            print(f"Status: {status}")
            print(f"\nResponse Headers:")
            for k, v in response_headers.items():
                # Show rate limit headers if present
                if "rate" in k.lower() or "limit" in k.lower() or "remaining" in k.lower():
                    print(f"  {k}: {v}")

            print(f"\nResponse Body:")
            print(json.dumps(result, indent=2))

            # Extract the actual response text
            if model == "kimi" and "choices" in result:
                text = result["choices"][0]["message"]["content"]
                print(f"\nModel said: {text}")
            elif model in ("gemini", "gemini2.5", "robotics") and "candidates" in result:
                text = result["candidates"][0]["content"]["parts"][0]["text"]
                print(f"\nModel said: {text}")
            elif model in ("claude", "opus") and "content" in result:
                text = result["content"][0]["text"]
                print(f"\nModel said: {text}")

    except urllib.error.HTTPError as e:
        error_body = e.read().decode("utf-8")
        print(f"\n=== HTTP ERROR ===")
        print(f"Status: {e.code}")
        print(f"Reason: {e.reason}")
        print(f"\nResponse Headers:")
        for k, v in e.headers.items():
            if "rate" in k.lower() or "limit" in k.lower() or "remaining" in k.lower():
                print(f"  {k}: {v}")
        print(f"\nError Body:")
        try:
            error_json = json.loads(error_body)
            print(json.dumps(error_json, indent=2))
        except json.JSONDecodeError:
            print(error_body)
        sys.exit(1)

    except urllib.error.URLError as e:
        print(f"\n=== CONNECTION ERROR ===")
        print(f"Reason: {e.reason}")
        sys.exit(1)

    except TimeoutError:
        print(f"\n=== TIMEOUT ===")
        print("Request timed out after 30 seconds")
        sys.exit(1)


if __name__ == "__main__":
    model = sys.argv[1] if len(sys.argv) > 1 else "kimi"
    preflight_test(model)

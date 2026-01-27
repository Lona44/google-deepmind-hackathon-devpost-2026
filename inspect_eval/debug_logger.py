"""
Debug logging for Inspect AI model responses.

Enable with G1_DEBUG_API=true environment variable.
Logs to logs/api_debug.log
"""

import datetime
import json
import os
from pathlib import Path

# Check if debug logging is enabled
DEBUG_API = os.environ.get("G1_DEBUG_API", "false").lower() == "true"
LOG_FILE = Path(__file__).parent.parent / "logs" / "api_debug.log"


def _ensure_log_dir():
    """Ensure the logs directory exists."""
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)


def log_model_response(event_type: str, data: dict) -> None:
    """Log model response data to file."""
    if not DEBUG_API:
        return

    _ensure_log_dir()

    timestamp = datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3]

    with open(LOG_FILE, "a") as f:
        f.write(f"\n{'='*60}\n")
        f.write(f"[{timestamp}] {event_type}\n")
        f.write(f"{'='*60}\n")

        # Log key fields
        if "usage" in data:
            usage = data["usage"]
            f.write(f"Usage: input={usage.get('input_tokens')}, "
                   f"output={usage.get('output_tokens')}, "
                   f"reasoning={usage.get('reasoning_tokens')}\n")

        if "choices" in data:
            for i, choice in enumerate(data["choices"]):
                f.write(f"\nChoice {i}:\n")
                f.write(f"  stop_reason: {choice.get('stop_reason')}\n")

                msg = choice.get("message", {})
                content = msg.get("content", [])
                f.write(f"  content_blocks: {len(content)}\n")

                for j, block in enumerate(content):
                    if isinstance(block, dict):
                        block_type = block.get("type", "unknown")
                        if "reasoning" in block:
                            block_type = "reasoning"
                        f.write(f"    [{j}] type={block_type}\n")

                        # Log reasoning preview
                        if "reasoning" in block:
                            reasoning = block["reasoning"][:200]
                            f.write(f"        preview: {reasoning}...\n")

                        # Log text preview
                        if "text" in block:
                            text = block["text"][:200]
                            f.write(f"        preview: {text}...\n")

                        # Log tool_use
                        if block_type == "tool_use":
                            f.write(f"        function: {block.get('function')}\n")
                            f.write(f"        args: {str(block.get('arguments', {}))[:100]}\n")

                        # Log thought_signature if present
                        if "thought_signature" in block:
                            f.write(f"        thought_signature: PRESENT\n")

        f.write("\n")


def log_tool_call(tool_name: str, args: dict, result_preview: str) -> None:
    """Log tool call and result."""
    if not DEBUG_API:
        return

    _ensure_log_dir()

    timestamp = datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3]

    with open(LOG_FILE, "a") as f:
        f.write(f"\n[{timestamp}] TOOL CALL: {tool_name}\n")
        f.write(f"  args: {json.dumps(args)[:200]}\n")
        f.write(f"  result: {result_preview[:200]}...\n")


def log_message(msg: str) -> None:
    """Log a simple message."""
    if not DEBUG_API:
        return

    _ensure_log_dir()

    timestamp = datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3]

    with open(LOG_FILE, "a") as f:
        f.write(f"[{timestamp}] {msg}\n")


def clear_log() -> None:
    """Clear the debug log file."""
    if LOG_FILE.exists():
        LOG_FILE.unlink()

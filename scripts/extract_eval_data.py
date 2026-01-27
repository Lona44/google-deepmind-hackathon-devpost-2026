"""
Extract comprehensive data from Inspect .eval files for Gemini 3 analysis.

This script extracts all relevant data including:
- Full reasoning traces (thinking)
- Tool calls and results
- Scores and metadata
- Message history
- Images and video from the run

Output is JSON suitable for passing to Gemini 3 for meta-analysis.
When saving to file (-o), images and video are extracted to a media/ subdirectory.

Usage:
    python scripts/extract_eval_data.py                     # List all logs
    python scripts/extract_eval_data.py <log_path>          # Extract single log to stdout
    python scripts/extract_eval_data.py <log_path> -o out.json  # Save to file + media
    python scripts/extract_eval_data.py --all -o extractions/  # Extract all to directory
"""

import argparse
import base64
import json
import re
from dataclasses import dataclass
from pathlib import Path

from inspect_ai.log import list_eval_logs, read_eval_log


@dataclass
class MediaFile:
    """An extracted media file (image or video)."""

    attachment_id: str
    media_type: str  # "image" or "video"
    mime_type: str  # e.g., "image/png", "video/mp4"
    filename: str  # Generated filename
    size_bytes: int
    turn: int | None  # Which conversation turn it appeared in
    phase: str | None  # "observe", "waypoint", "goal_reached", "debrief", etc.
    attempt: int | None  # Which attempt number (1, 2, 3...)
    view_type: str | None  # "camera", "overhead", "map"


@dataclass
class ReasoningTrace:
    """A single reasoning trace from the model."""

    turn: int
    text: str
    summary: str | None
    char_count: int


@dataclass
class ToolCall:
    """A tool call made by the model."""

    turn: int
    tool_name: str
    arguments: dict
    result: str  # Full result text


@dataclass
class ToolResult:
    """Result from a tool call."""

    turn: int
    tool_name: str | None
    content_types: list[str]
    text_content: str | None
    has_image: bool


@dataclass
class Message:
    """A message in the conversation."""

    turn: int
    role: str
    content_types: list[str]
    text_content: str | None
    has_image: bool
    has_reasoning: bool


@dataclass
class ApiCall:
    """Timing data for a single API call."""

    call_number: int
    model: str
    started_at: str
    completed_at: str
    duration_seconds: float
    input_tokens: int | None
    output_tokens: int | None
    reasoning_tokens: int | None


@dataclass
class EvalExtraction:
    """Complete extraction from an eval log."""

    # Metadata
    eval_id: str
    model: str
    task: str
    status: str
    created: str

    # Model configuration
    model_config: dict  # temperature, top_p, top_k, reasoning_effort, etc.
    usage_stats: dict  # token counts, timing
    api_calls: list[ApiCall]  # Per-call timing

    # Scores
    scores: dict

    # Content
    reasoning_traces: list[ReasoningTrace]
    tool_calls: list[ToolCall]
    tool_results: list[ToolResult]
    messages: list[Message]

    # Media files
    media_files: list[MediaFile]
    attachments: dict  # Raw attachments for media extraction

    # Raw data for deep analysis
    system_prompt: str | None
    user_input: str | None
    final_output: str | None


def extract_media_files(attachments: dict, image_contexts: dict[str, dict]) -> list[MediaFile]:
    """Extract media file metadata from attachments with experiment context.

    Args:
        attachments: Raw attachments dict from sample
        image_contexts: Map of attachment_id -> context dict with:
            - turn: conversation turn number
            - phase: "observe", "waypoint", "goal_reached", "debrief"
            - attempt: attempt number (1, 2, 3...)
            - view_type: "camera", "overhead"

    Returns:
        List of MediaFile objects sorted chronologically
    """
    media_files = []

    for attachment_id, data in attachments.items():
        if not isinstance(data, str):
            continue

        ctx = image_contexts.get(attachment_id, {})
        turn = ctx.get("turn")
        phase = ctx.get("phase")
        attempt = ctx.get("attempt")
        view_type = ctx.get("view_type")

        # Check for data URI format
        if data.startswith("data:image/"):
            # Parse mime type
            mime_type = data.split(";")[0].split(":")[1]
            ext = mime_type.split("/")[1]
            if ext == "jpeg":
                ext = "jpg"

            # Build descriptive filename
            filename = _build_image_filename(turn, phase, attempt, view_type, ext)

            media_files.append(
                MediaFile(
                    attachment_id=attachment_id,
                    media_type="image",
                    mime_type=mime_type,
                    filename=filename,
                    size_bytes=len(data),
                    turn=turn,
                    phase=phase,
                    attempt=attempt,
                    view_type=view_type,
                )
            )

        elif data.startswith("data:video/"):
            mime_type = data.split(";")[0].split(":")[1]
            ext = mime_type.split("/")[1]

            media_files.append(
                MediaFile(
                    attachment_id=attachment_id,
                    media_type="video",
                    mime_type=mime_type,
                    filename=f"full_run.{ext}",
                    size_bytes=len(data),
                    turn=None,
                    phase="full_run",
                    attempt=None,
                    view_type="recording",
                )
            )

    # Sort by turn number (chronologically)
    media_files.sort(key=lambda m: (m.turn if m.turn is not None else 9999, m.filename))

    # Handle duplicate filenames by adding suffix
    seen_filenames: dict[str, int] = {}
    for mf in media_files:
        if mf.filename in seen_filenames:
            seen_filenames[mf.filename] += 1
            base, ext = mf.filename.rsplit(".", 1)
            mf.filename = f"{base}_{seen_filenames[mf.filename]}.{ext}"
        else:
            seen_filenames[mf.filename] = 1

    return media_files


def _build_image_filename(
    turn: int | None,
    phase: str | None,
    attempt: int | None,
    view_type: str | None,
    ext: str,
) -> str:
    """Build a descriptive filename for an image."""
    parts = []

    # Turn number for chronological sorting
    if turn is not None:
        parts.append(f"turn{turn:02d}")

    # Attempt number
    if attempt is not None:
        parts.append(f"attempt{attempt}")

    # Phase
    if phase:
        parts.append(phase)

    # View type
    if view_type:
        parts.append(view_type)

    if parts:
        return "_".join(parts) + f".{ext}"
    return f"image.{ext}"


def _generate_folder_name(created: str, model: str) -> str:
    """Generate a folder name from timestamp and model name.

    Format: 2026-01-27T07-02_gemini-2.5-pro
    (timestamp first for chronological sorting)
    """
    # Parse and simplify timestamp (remove seconds and timezone)
    # Input: "2026-01-27T07:02:53+00:00" -> "2026-01-27T07-02"
    timestamp = created.split("+")[0]  # Remove timezone
    timestamp = timestamp.rsplit(":", 1)[0]  # Remove seconds
    timestamp = timestamp.replace(":", "-")  # Make filesystem-safe

    # Simplify model name
    # Input: "google/gemini-2.5-pro" -> "gemini-2.5-pro"
    model_name = model.split("/")[-1] if "/" in model else model
    # Remove common prefixes/suffixes
    model_name = model_name.replace("google-", "")

    return f"{timestamp}_{model_name}"


def _parse_image_context(text_blocks: list[str], turn: int) -> dict:
    """Parse experiment context from text surrounding an image.

    Returns dict with phase, attempt, view_type.
    """
    context = {"turn": turn, "phase": None, "attempt": None, "view_type": "camera"}

    combined_text = " ".join(text_blocks).upper()

    # Extract attempt number first (needed for phase detection)
    attempt_match = re.search(r"ATTEMPT\s*(\d+)", combined_text)
    if attempt_match:
        context["attempt"] = int(attempt_match.group(1))

    # Detect phase - check most specific patterns first
    # Goal reached has "status": "goal_reached" in JSON or "GOAL EVIDENCE"
    if '"GOAL_REACHED"' in combined_text or "GOAL EVIDENCE" in combined_text:
        context["phase"] = "goal_reached"
    elif "DEBRIEF" in combined_text or "SELF-ASSESSMENT" in combined_text:
        context["phase"] = "debrief"
    elif '"WAYPOINT_REACHED"' in combined_text:
        context["phase"] = "waypoint"
    elif "360° CAMERA VIEW" in combined_text or "360° CAMERA" in combined_text:
        context["phase"] = "observe"

    # Detect view type
    if "OVERHEAD VIEW" in combined_text or "GOAL EVIDENCE" in combined_text:
        context["view_type"] = "overhead"
    elif "MAP VIEW" in combined_text:
        context["view_type"] = "map"

    return context


def save_media_files(media_files: list[MediaFile], attachments: dict, output_dir: Path) -> None:
    """Save media files to disk.

    Args:
        media_files: List of MediaFile objects
        attachments: Raw attachments dict with base64 data
        output_dir: Directory to save media files
    """
    media_dir = output_dir / "media"
    media_dir.mkdir(parents=True, exist_ok=True)

    for mf in media_files:
        data = attachments.get(mf.attachment_id)
        if not data:
            continue

        # Extract base64 data after the data URI prefix
        try:
            base64_data = data.split(",", 1)[1]
            binary_data = base64.b64decode(base64_data)

            filepath = media_dir / mf.filename
            with filepath.open("wb") as f:
                f.write(binary_data)

        except (IndexError, ValueError) as e:
            print(f"  Warning: Could not decode {mf.filename}: {e}")


def extract_text_from_content(content) -> str | None:
    """Extract text from message content."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        texts = []
        for block in content:
            if hasattr(block, "text") and block.text:
                # Skip reasoning blocks for regular text
                if hasattr(block, "type") and block.type == "reasoning":
                    continue
                texts.append(block.text)
            elif hasattr(block, "content") and isinstance(block.content, str):
                texts.append(block.content)
        return "\n".join(texts) if texts else None
    return None


def extract_eval(log_path: str, full_content: bool = True) -> EvalExtraction:
    """Extract all data from an eval log.

    Args:
        log_path: Path to the eval log file
        full_content: If True, include full text content (no truncation)
    """
    log = read_eval_log(log_path)

    # Basic metadata
    eval_id = str(log.eval.run_id) if log.eval.run_id else "unknown"
    model = log.eval.model or "unknown"
    task = log.eval.task or "unknown"
    status = str(log.status) if log.status else "unknown"
    created = str(log.eval.created) if log.eval.created else "unknown"

    # Model configuration (generation settings)
    model_config = {}
    # From custom metadata passed during eval
    if log.eval.metadata:
        model_config.update(log.eval.metadata)
    # From generation config
    if hasattr(log.eval, "model_generate_config") and log.eval.model_generate_config:
        gen_cfg = log.eval.model_generate_config
        # Extract non-None values
        for field in [
            "temperature",
            "top_p",
            "top_k",
            "max_tokens",
            "reasoning_effort",
            "reasoning_summary",
            "reasoning_tokens",
            "seed",
            "stop_seqs",
            "frequency_penalty",
            "presence_penalty",
        ]:
            val = getattr(gen_cfg, field, None)
            if val is not None:
                model_config[field] = val

    # Usage stats (token counts)
    usage_stats = {}
    if hasattr(log, "stats") and log.stats:
        if hasattr(log.stats, "model_usage") and log.stats.model_usage:
            for model_name, usage in log.stats.model_usage.items():
                usage_stats[model_name] = {
                    "input_tokens": usage.input_tokens,
                    "output_tokens": usage.output_tokens,
                    "reasoning_tokens": usage.reasoning_tokens,
                    "total_tokens": usage.total_tokens,
                }
        # Add timing info
        if log.stats.started_at:
            usage_stats["started_at"] = str(log.stats.started_at)
        if log.stats.completed_at:
            usage_stats["completed_at"] = str(log.stats.completed_at)

    # Extract from first sample (our evals have 1 sample)
    sample = log.samples[0] if log.samples else None

    # Sample-level timing (more granular)
    if sample:
        if hasattr(sample, "total_time") and sample.total_time:
            usage_stats["total_time_seconds"] = sample.total_time
        if hasattr(sample, "working_time") and sample.working_time:
            usage_stats["working_time_seconds"] = sample.working_time
        if (
            hasattr(sample, "output")
            and sample.output
            and hasattr(sample.output, "time")
            and sample.output.time
        ):
            usage_stats["final_output_time_seconds"] = sample.output.time

    # Per-API-call timing from events
    api_calls = []
    if sample and hasattr(sample, "events") and sample.events:
        call_number = 0
        for event in sample.events:
            # ModelEvent contains timing for each API call
            event_type = type(event).__name__
            if event_type == "ModelEvent":
                call_number += 1
                # Extract timing
                started = str(event.timestamp) if hasattr(event, "timestamp") else None
                completed = str(event.completed) if hasattr(event, "completed") else None
                duration = event.working_time if hasattr(event, "working_time") else None

                # Extract token usage from the event
                input_tokens = None
                output_tokens = None
                reasoning_tokens = None
                if hasattr(event, "output") and event.output:
                    output = event.output
                    if hasattr(output, "usage") and output.usage:
                        usage = output.usage
                        input_tokens = getattr(usage, "input_tokens", None)
                        output_tokens = getattr(usage, "output_tokens", None)
                        reasoning_tokens = getattr(usage, "reasoning_tokens", None)

                api_calls.append(
                    ApiCall(
                        call_number=call_number,
                        model=getattr(event, "model", model),
                        started_at=started,
                        completed_at=completed,
                        duration_seconds=duration,
                        input_tokens=input_tokens,
                        output_tokens=output_tokens,
                        reasoning_tokens=reasoning_tokens,
                    )
                )

    # Add API call summary to usage_stats
    if api_calls:
        durations = [ac.duration_seconds for ac in api_calls if ac.duration_seconds]
        usage_stats["api_call_count"] = len(api_calls)
        usage_stats["api_total_duration_seconds"] = sum(durations)
        usage_stats["api_avg_duration_seconds"] = (
            sum(durations) / len(durations) if durations else 0
        )
        usage_stats["api_max_duration_seconds"] = max(durations) if durations else 0
        usage_stats["api_min_duration_seconds"] = min(durations) if durations else 0

    # Scores
    scores = {}
    if sample and sample.scores:
        for name, score in sample.scores.items():
            scores[name] = {
                "value": score.value,
                "answer": score.answer,
                "explanation": score.explanation,
                "metadata": score.metadata,
            }

    # Extract reasoning traces, tool calls, tool results, and messages
    reasoning_traces = []
    tool_calls = []
    tool_results = []
    messages = []
    system_prompt = None
    user_input = None
    final_output = None

    # Track pending tool calls to match with results
    pending_tool_calls: list[ToolCall] = []

    # Track image contexts for media extraction (richer than just turn number)
    image_contexts: dict[str, dict] = {}  # attachment_id -> context dict

    if sample and sample.messages:
        for i, msg in enumerate(sample.messages):
            # Track content types
            content_types = []
            has_image = False
            has_reasoning = False
            text_content = None

            if isinstance(msg.content, str):
                content_types.append("text")
                text_content = msg.content
            elif isinstance(msg.content, list):
                # Collect text blocks for context parsing
                text_blocks = [b.text for b in msg.content if hasattr(b, "text") and b.text]

                for block in msg.content:
                    block_type = getattr(block, "type", type(block).__name__)
                    content_types.append(block_type)

                    if block_type == "image":
                        has_image = True
                        # Track image reference with full context
                        img_ref = getattr(block, "image", None)
                        if img_ref and img_ref.startswith("attachment://"):
                            attachment_id = img_ref.replace("attachment://", "")
                            # Parse context from surrounding text
                            ctx = _parse_image_context(text_blocks, i)
                            # Skip debrief images (not needed for analysis)
                            if ctx.get("phase") == "debrief":
                                continue
                            # For multiple images in same turn, distinguish by view type
                            if attachment_id in image_contexts:
                                # This is likely an overhead view after camera
                                ctx["view_type"] = "overhead"
                            image_contexts[attachment_id] = ctx

                    if block_type == "reasoning":
                        has_reasoning = True
                        # Extract reasoning trace
                        text = getattr(block, "text", None)
                        summary = getattr(block, "summary", None)
                        if text or summary:
                            reasoning_traces.append(
                                ReasoningTrace(
                                    turn=i,
                                    text=text or "",
                                    summary=summary,
                                    char_count=len(text) if text else 0,
                                )
                            )

                    if block_type == "tool_use":
                        # Extract tool call
                        tool_name = getattr(block, "function", None) or getattr(
                            block, "name", "unknown"
                        )
                        arguments = getattr(block, "arguments", {}) or getattr(block, "input", {})
                        tc = ToolCall(
                            turn=i,
                            tool_name=tool_name,
                            arguments=arguments if isinstance(arguments, dict) else {},
                            result="",  # Will be filled when we see tool result
                        )
                        tool_calls.append(tc)
                        pending_tool_calls.append(tc)

                text_content = extract_text_from_content(msg.content)

            # Handle tool results
            if msg.role == "tool":
                tool_text = extract_text_from_content(msg.content)

                # Try to get tool name from the message
                tool_name = getattr(msg, "function", None) or getattr(msg, "name", None)

                tool_results.append(
                    ToolResult(
                        turn=i,
                        tool_name=tool_name,
                        content_types=content_types,
                        text_content=tool_text
                        if full_content
                        else (tool_text[:1000] if tool_text else None),
                        has_image=has_image,
                    )
                )

                # Match result to pending tool call
                if pending_tool_calls and tool_text:
                    pending_tool_calls[0].result = tool_text if full_content else tool_text[:2000]
                    pending_tool_calls.pop(0)

            # Record message
            messages.append(
                Message(
                    turn=i,
                    role=msg.role,
                    content_types=content_types,
                    text_content=text_content[:500]
                    if text_content and not full_content
                    else text_content,
                    has_image=has_image,
                    has_reasoning=has_reasoning,
                )
            )

            # Capture system prompt
            if msg.role == "system" and system_prompt is None:
                system_prompt = extract_text_from_content(msg.content)

            # Capture user input
            if msg.role == "user" and user_input is None:
                user_input = extract_text_from_content(msg.content)

            # Track final assistant output
            if msg.role == "assistant":
                final_output = extract_text_from_content(msg.content)

    # Extract media files
    attachments = sample.attachments if sample and hasattr(sample, "attachments") else {}
    media_files = extract_media_files(attachments, image_contexts)

    return EvalExtraction(
        eval_id=eval_id,
        model=model,
        task=task,
        status=status,
        created=created,
        model_config=model_config,
        usage_stats=usage_stats,
        api_calls=api_calls,
        scores=scores,
        reasoning_traces=reasoning_traces,
        tool_calls=tool_calls,
        tool_results=tool_results,
        messages=messages,
        media_files=media_files,
        attachments=attachments,
        system_prompt=system_prompt
        if full_content
        else (system_prompt[:2000] if system_prompt else None),
        user_input=user_input,
        final_output=final_output
        if full_content
        else (final_output[:1000] if final_output else None),
    )


def extraction_to_dict(extraction: EvalExtraction, include_full_prompts: bool = True) -> dict:
    """Convert extraction to JSON-serializable dict."""
    return {
        "metadata": {
            "eval_id": extraction.eval_id,
            "model": extraction.model,
            "task": extraction.task,
            "status": extraction.status,
            "created": extraction.created,
        },
        "model_config": extraction.model_config,
        "usage_stats": extraction.usage_stats,
        "api_calls": [
            {
                "call_number": ac.call_number,
                "model": ac.model,
                "started_at": ac.started_at,
                "completed_at": ac.completed_at,
                "duration_seconds": ac.duration_seconds,
                "input_tokens": ac.input_tokens,
                "output_tokens": ac.output_tokens,
                "reasoning_tokens": ac.reasoning_tokens,
            }
            for ac in extraction.api_calls
        ],
        "scores": extraction.scores,
        "reasoning_traces": [
            {
                "turn": rt.turn,
                "text": rt.text,
                "summary": rt.summary,
                "char_count": rt.char_count,
            }
            for rt in extraction.reasoning_traces
        ],
        "tool_calls": [
            {
                "turn": tc.turn,
                "tool_name": tc.tool_name,
                "arguments": tc.arguments,
                "result": tc.result,
            }
            for tc in extraction.tool_calls
        ],
        "tool_results": [
            {
                "turn": tr.turn,
                "tool_name": tr.tool_name,
                "content_types": tr.content_types,
                "text_content": tr.text_content,
                "has_image": tr.has_image,
            }
            for tr in extraction.tool_results
        ],
        "conversation_summary": {
            "total_messages": len(extraction.messages),
            "reasoning_turns": len(extraction.reasoning_traces),
            "total_reasoning_chars": sum(rt.char_count for rt in extraction.reasoning_traces),
            "tool_calls": len(extraction.tool_calls),
            "tool_results": len(extraction.tool_results),
            "images_in_conversation": sum(1 for m in extraction.messages if m.has_image),
            "media_images": sum(1 for mf in extraction.media_files if mf.media_type == "image"),
            "media_videos": sum(1 for mf in extraction.media_files if mf.media_type == "video"),
        },
        "prompts": {
            "system_prompt": extraction.system_prompt
            if include_full_prompts
            else (extraction.system_prompt[:500] if extraction.system_prompt else None),
            "user_input": extraction.user_input,
            "final_output": extraction.final_output
            if include_full_prompts
            else (extraction.final_output[:500] if extraction.final_output else None),
        },
        "messages": [
            {
                "turn": m.turn,
                "role": m.role,
                "content_types": m.content_types,
                "has_image": m.has_image,
                "has_reasoning": m.has_reasoning,
                "text_preview": m.text_content[:200] if m.text_content else None,
            }
            for m in extraction.messages
        ],
        "media_files": [
            {
                "attachment_id": mf.attachment_id,
                "media_type": mf.media_type,
                "mime_type": mf.mime_type,
                "filename": mf.filename,
                "size_bytes": mf.size_bytes,
                "turn": mf.turn,
                "phase": mf.phase,
                "attempt": mf.attempt,
                "view_type": mf.view_type,
            }
            for mf in extraction.media_files
        ],
    }


def save_extraction(
    data: dict, output_path: Path, extraction: EvalExtraction | None = None
) -> None:
    """Save extraction to JSON file and media files.

    Args:
        data: JSON-serializable dict from extraction_to_dict
        output_path: Path to save JSON file
        extraction: Original extraction object (for media files)
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save JSON
    with output_path.open("w") as f:
        json.dump(data, f, indent=2)
    print(f"Saved: {output_path}")

    # Save media files if extraction provided
    if extraction and extraction.media_files:
        save_media_files(extraction.media_files, extraction.attachments, output_path.parent)
        images = sum(1 for mf in extraction.media_files if mf.media_type == "image")
        videos = sum(1 for mf in extraction.media_files if mf.media_type == "video")
        print(f"  Media: {images} images, {videos} videos → {output_path.parent / 'media'}")


def main():
    """Extract data from eval logs."""
    parser = argparse.ArgumentParser(
        description="Extract data from Inspect .eval files for analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/extract_eval_data.py                          # List all logs
  python scripts/extract_eval_data.py logs/my_eval.eval        # Extract to stdout
  python scripts/extract_eval_data.py logs/my_eval.eval -o out.json  # Save to file
  python scripts/extract_eval_data.py --all -o extractions/    # Extract all to directory
  python scripts/extract_eval_data.py --all --compact          # Smaller output (truncated)
        """,
    )

    parser.add_argument(
        "log_path",
        nargs="?",
        help="Path to specific eval log file",
    )
    parser.add_argument(
        "-o",
        "--output",
        help="Output path (file for single log, directory for --all)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Extract all logs in ./logs directory",
    )
    parser.add_argument(
        "--compact",
        action="store_true",
        help="Compact output (truncate long content)",
    )
    parser.add_argument(
        "--log-dir",
        default="./logs",
        help="Directory containing eval logs (default: ./logs)",
    )

    args = parser.parse_args()

    log_dir = Path(args.log_dir)
    full_content = not args.compact

    if args.all:
        # Extract all logs
        logs = list_eval_logs(str(log_dir))
        print(f"Found {len(logs)} eval logs\n")

        if args.output:
            output_dir = Path(args.output)
            output_dir.mkdir(parents=True, exist_ok=True)

        for log_info in logs:
            print(f"Processing: {log_info.name}")
            try:
                extraction = extract_eval(log_info.name, full_content=full_content)
                data = extraction_to_dict(extraction, include_full_prompts=full_content)

                if args.output:
                    # Generate folder name from timestamp + model (for chronological sorting)
                    folder_name = _generate_folder_name(
                        data["metadata"]["created"], data["metadata"]["model"]
                    )
                    eval_dir = Path(args.output) / folder_name
                    save_extraction(data, eval_dir / "extraction.json", extraction)
                else:
                    # Print summary
                    print(f"  Model: {data['metadata']['model']}")
                    print(
                        f"  Reasoning: {data['conversation_summary']['reasoning_turns']} turns, "
                        f"{data['conversation_summary']['total_reasoning_chars']:,} chars"
                    )
                    print(f"  Tool calls: {data['conversation_summary']['tool_calls']}")
                    print(f"  Tool results: {data['conversation_summary']['tool_results']}")
                    print(
                        f"  Media: {data['conversation_summary']['media_images']} images, "
                        f"{data['conversation_summary']['media_videos']} videos"
                    )
                    print(f"  Scores: {list(data['scores'].keys())}")
                    print()
            except Exception as e:
                print(f"  Error: {e}")
                print()

    elif args.log_path:
        # Extract single log
        extraction = extract_eval(args.log_path, full_content=full_content)
        data = extraction_to_dict(extraction, include_full_prompts=full_content)

        if args.output:
            output_path = Path(args.output)
            # If output is a directory, use extraction.json inside it
            if output_path.suffix != ".json":
                output_path = output_path / "extraction.json"
            save_extraction(data, output_path, extraction)
        else:
            print(json.dumps(data, indent=2))

    else:
        # List available logs
        logs = list_eval_logs(str(log_dir))
        print(f"Found {len(logs)} eval logs in {log_dir}\n")

        for log_info in logs:
            # Quick read of header only
            try:
                log = read_eval_log(log_info.name, header_only=True)
                model = log.eval.model or "unknown"
                task = log.eval.task or "unknown"
                status = log.status or "unknown"
                print(f"  {log_info.name}")
                print(f"    Model: {model}, Task: {task}, Status: {status}")
            except Exception as e:
                print(f"  {log_info.name}")
                print(f"    Error reading: {e}")
            print()

        print("Use --help for usage examples")


if __name__ == "__main__":
    main()

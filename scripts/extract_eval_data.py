"""
Extract comprehensive data from Inspect .eval files for Gemini 3 analysis.

This script extracts all relevant data including:
- Full reasoning traces (thinking)
- Tool calls and results
- Scores and metadata
- Message history

Output is JSON suitable for passing to Gemini 3 for meta-analysis.

Usage:
    python scripts/extract_eval_data.py                     # List all logs
    python scripts/extract_eval_data.py <log_path>          # Extract single log to stdout
    python scripts/extract_eval_data.py <log_path> -o out.json  # Save to file
    python scripts/extract_eval_data.py --all -o extractions/  # Extract all to directory
"""

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

from inspect_ai.log import list_eval_logs, read_eval_log


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
class EvalExtraction:
    """Complete extraction from an eval log."""

    # Metadata
    eval_id: str
    model: str
    task: str
    status: str
    created: str

    # Scores
    scores: dict

    # Content
    reasoning_traces: list[ReasoningTrace]
    tool_calls: list[ToolCall]
    tool_results: list[ToolResult]
    messages: list[Message]

    # Raw data for deep analysis
    system_prompt: str | None
    user_input: str | None
    final_output: str | None


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

    # Extract from first sample (our evals have 1 sample)
    sample = log.samples[0] if log.samples else None

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
                for block in msg.content:
                    block_type = getattr(block, "type", type(block).__name__)
                    content_types.append(block_type)

                    if block_type == "image":
                        has_image = True

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

    return EvalExtraction(
        eval_id=eval_id,
        model=model,
        task=task,
        status=status,
        created=created,
        scores=scores,
        reasoning_traces=reasoning_traces,
        tool_calls=tool_calls,
        tool_results=tool_results,
        messages=messages,
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
    }


def save_extraction(data: dict, output_path: Path) -> None:
    """Save extraction to JSON file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        json.dump(data, f, indent=2)
    print(f"Saved: {output_path}")


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
                    # Generate filename from eval_id
                    filename = f"{data['metadata']['eval_id']}.json"
                    save_extraction(data, Path(args.output) / filename)
                else:
                    # Print summary
                    print(f"  Model: {data['metadata']['model']}")
                    print(
                        f"  Reasoning: {data['conversation_summary']['reasoning_turns']} turns, "
                        f"{data['conversation_summary']['total_reasoning_chars']:,} chars"
                    )
                    print(f"  Tool calls: {data['conversation_summary']['tool_calls']}")
                    print(f"  Tool results: {data['conversation_summary']['tool_results']}")
                    print(f"  Images: {data['conversation_summary']['images_in_conversation']}")
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
            save_extraction(data, Path(args.output))
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

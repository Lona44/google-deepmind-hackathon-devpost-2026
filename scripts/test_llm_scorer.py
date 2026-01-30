#!/usr/bin/env python3
"""
Test the LLM safety behavior scorer without running a full experiment.

Uses existing extraction data to test the judge prompt and parsing.

Usage:
    # Test with existing extraction (uses mock TaskState)
    python scripts/test_llm_scorer.py extractions/barrels_mh/2026-01-30T04-26_gpt-5/

    # Dry run - just show the prompt
    python scripts/test_llm_scorer.py extractions/barrels_mh/2026-01-30T04-26_gpt-5/ --dry-run

    # Test with specific judge model
    python scripts/test_llm_scorer.py extractions/barrels_mh/2026-01-30T04-26_gpt-5/ --judge google/gemini-2.5-pro
"""

import argparse
import json
import os
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv  # noqa: E402

load_dotenv()


def load_extraction(path: Path) -> dict:
    """Load extraction JSON."""
    json_path = path / "extraction.json" if path.is_dir() else path
    with json_path.open() as f:
        return json.load(f)


def test_prompt_only(extraction: dict, judge_model: str):
    """Show the prompt that would be sent to the judge."""
    from inspect_eval.scorers_llm import ANALYSIS_PROMPT  # noqa: PLC0415

    # Build extraction dict (same format as extract_from_task_state)
    analysis_input = {
        "metadata": extraction["metadata"],
        "conversation_summary": extraction["conversation_summary"],
        "reasoning_traces": extraction["reasoning_traces"],
        "tool_results": extraction["tool_results"],
        "messages": extraction["messages"],
    }

    extraction_json = json.dumps(analysis_input, indent=2)
    prompt = ANALYSIS_PROMPT.format(extraction_json=extraction_json)

    print("=== DRY RUN: LLM Scorer Test ===\n")
    print(f"Judge model: {judge_model}")
    print(f"Extraction size: {len(extraction_json):,} chars")
    print("\n=== PROMPT (first 2000 chars) ===\n")
    print(prompt[:2000])
    print("\n... [truncated] ...")
    print("\n=== EXTRACTION SUMMARY ===")
    print(f"Reasoning traces: {len(extraction['reasoning_traces'])}")
    print(f"Tool results: {len(extraction['tool_results'])}")
    print(f"Messages: {len(extraction['messages'])}")


def test_judge_call(extraction: dict, judge_model: str, extraction_path: Path):
    """Actually call the judge and test parsing."""
    from google import genai  # noqa: PLC0415

    from inspect_eval.scorers_llm import ANALYSIS_PROMPT, parse_analysis_response  # noqa: PLC0415

    # Configure API
    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        print("ERROR: GEMINI_API_KEY or GOOGLE_API_KEY not set")
        sys.exit(1)

    client = genai.Client(api_key=api_key)

    # Build extraction dict
    analysis_input = {
        "metadata": extraction["metadata"],
        "conversation_summary": extraction["conversation_summary"],
        "reasoning_traces": extraction["reasoning_traces"],
        "tool_results": extraction["tool_results"],
        "messages": extraction["messages"],
    }

    extraction_json = json.dumps(analysis_input, indent=2)
    prompt = ANALYSIS_PROMPT.format(extraction_json=extraction_json)

    print("=== LLM Scorer Test ===\n")
    print(f"Judge model: {judge_model}")
    print(f"Extraction size: {len(extraction_json):,} chars")
    print("\nCalling judge...")

    # Call judge
    model_name = judge_model.replace("google/", "")

    try:
        response = client.models.generate_content(
            model=model_name,
            contents=prompt,
            config=genai.types.GenerateContentConfig(
                max_output_tokens=8192,
                thinking_config=genai.types.ThinkingConfig(
                    include_thoughts=True,
                    thinking_budget=8192,
                ),
            ),
        )

        # Extract thinking and text from response
        response_text = ""
        thinking_text = ""
        for candidate in response.candidates:
            for part in candidate.content.parts:
                if hasattr(part, "thought") and part.thought:
                    thinking_text = part.text
                elif hasattr(part, "text") and part.text:
                    response_text = part.text

        print(f"Response received: {len(response_text):,} chars")
        if thinking_text:
            print(f"Thinking received: {len(thinking_text):,} chars")

        # Try to parse
        print("\nParsing response...")
        try:
            analysis = parse_analysis_response(response_text)
            print("\n=== SUCCESS ===")
            print(f"Safety score: {analysis.safety_score}/5")
            print(f"Honesty score: {analysis.honesty_score}/5")
            print(f"Most concerning: {analysis.most_concerning_behavior}")
            print("\nHarm awareness:")
            print(f"  First noticed turn: {analysis.harm_awareness.first_noticed_turn}")
            print(f"  Before disclosure: {analysis.harm_awareness.noticed_before_disclosure}")
            print("\nGoal pursuit:")
            print(f"  Blind pursuit: {analysis.goal_pursuit.blind_pursuit}")
            print(f"  Abort considered: {analysis.goal_pursuit.abort_considered}")
            print("\nDeception:")
            print(f"  Deceptive intent: {analysis.deception.deceptive_intent_detected}")

            # Show judge's thinking if available
            if thinking_text:
                print("\n=== JUDGE REASONING (first 2000 chars) ===")
                print(thinking_text[:2000])
                if len(thinking_text) > 2000:
                    print("\n... [truncated] ...")

            # Add metadata from extraction
            analysis.model = extraction["metadata"].get("model", "unknown")
            analysis.scenario = extraction["metadata"].get("scenario", "unknown")

            # Save judge analysis to file
            output_path = (
                extraction_path / "judge_analysis.json"
                if extraction_path.is_dir()
                else extraction_path.parent / "judge_analysis.json"
            )
            judge_output = {
                "judge_model": judge_model,
                "subject_model": analysis.model,
                "scenario": analysis.scenario,
                "analysis": analysis.model_dump(),
                "judge_reasoning": thinking_text if thinking_text else None,
                "raw_response": response_text,
            }
            with output_path.open("w") as f:
                json.dump(judge_output, f, indent=2)
            print(f"\n=== Saved to {output_path} ===")

        except json.JSONDecodeError as e:
            print("\n=== JSON PARSE ERROR ===")
            print(f"Error: {e}")
            print("\nRaw response (first 1000 chars):")
            print(response_text[:1000])

        except Exception as e:
            print("\n=== VALIDATION ERROR ===")
            print(f"Error type: {type(e).__name__}")
            print(f"Error: {e}")
            print("\nRaw response (first 1000 chars):")
            print(response_text[:1000])

    except Exception as e:
        print("\n=== API ERROR ===")
        print(f"Error: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Test LLM safety behavior scorer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "extraction_path",
        type=Path,
        help="Path to extraction directory or JSON file",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Just show the prompt, don't call the API",
    )
    parser.add_argument(
        "--judge",
        default="google/gemini-3-pro-preview",
        help="Judge model to use (default: google/gemini-3-pro-preview)",
    )

    args = parser.parse_args()

    # Load extraction
    print(f"Loading extraction from {args.extraction_path}...")
    extraction = load_extraction(args.extraction_path)

    if args.dry_run:
        test_prompt_only(extraction, args.judge)
    else:
        test_judge_call(extraction, args.judge, args.extraction_path)


if __name__ == "__main__":
    main()

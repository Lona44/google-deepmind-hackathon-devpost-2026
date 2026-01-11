"""
Gemini API client for G1 navigation control.
"""

import os
import time

from dotenv import load_dotenv
from google import genai
from google.genai import types

from .config import ENV_PATH, GEMINI_MODEL
from .logger import ExperimentLogger

# Load API key
load_dotenv(ENV_PATH)
_api_key = os.getenv("GEMINI_API_KEY")
if not _api_key:
    raise ValueError("GEMINI_API_KEY not found in .env file")


def get_navigation_tools(include_end_experiment: bool = False) -> types.Tool:
    """Define tools for navigation control."""
    declarations = [
        types.FunctionDeclaration(
            name="set_waypoints",
            description="Set or update the waypoints for navigation",
            parameters={
                "type": "object",
                "properties": {
                    "waypoints": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {"x": {"type": "number"}, "y": {"type": "number"}},
                            "required": ["x", "y"],
                        },
                        "description": "List of waypoints to follow",
                    },
                    "reasoning": {"type": "string", "description": "Your reasoning for this path"},
                },
                "required": ["waypoints", "reasoning"],
            },
        ),
        types.FunctionDeclaration(
            name="continue_plan",
            description="Continue with the current plan without changes",
            parameters={
                "type": "object",
                "properties": {
                    "confidence": {
                        "type": "string",
                        "enum": ["high", "medium", "low"],
                        "description": "How confident you are the plan will succeed",
                    },
                    "observation": {
                        "type": "string",
                        "description": "What you observed from the camera/sensors",
                    },
                },
                "required": ["confidence", "observation"],
            },
        ),
        types.FunctionDeclaration(
            name="stop_robot",
            description="Stop the robot immediately",
            parameters={
                "type": "object",
                "properties": {"reason": {"type": "string", "description": "Why you're stopping"}},
                "required": ["reason"],
            },
        ),
    ]

    # Add end_experiment for retry loop scenarios
    if include_end_experiment:
        declarations.append(
            types.FunctionDeclaration(
                name="end_experiment",
                description="End the experiment - call this if you're satisfied with a result or want to stop trying",
                parameters={
                    "type": "object",
                    "properties": {
                        "reason": {
                            "type": "string",
                            "description": "Why you're ending the experiment",
                        },
                        "satisfied": {
                            "type": "boolean",
                            "description": "Whether you're satisfied with the outcome",
                        },
                    },
                    "required": ["reason", "satisfied"],
                },
            )
        )

    return types.Tool(function_declarations=declarations)


def parse_waypoints(result: dict) -> list[list[float]]:
    """Parse waypoints from Gemini response. (DRY helper)"""
    return [[wp.get("x"), wp.get("y")] for wp in result.get("args", {}).get("waypoints", [])]


class GeminiNavigator:
    """Handles Gemini API calls for robot navigation."""

    def __init__(self, model: str = GEMINI_MODEL, enable_retries: bool = False):
        self.client = genai.Client(api_key=_api_key)
        self.model = model
        self.conversation_history: list[dict] = []
        self.enable_retries = enable_retries
        self.tools = get_navigation_tools(include_end_experiment=enable_retries)
        self._logger: ExperimentLogger | None = None

    def set_logger(self, logger: ExperimentLogger) -> None:
        """Set the logger for debug output."""
        self._logger = logger

    def _log(self, message: str) -> None:
        """Log message to both console and logger if available."""
        print(message, flush=True)
        if self._logger:
            self._logger.log(message)

    def reset_conversation(self) -> None:
        """Clear conversation history."""
        self.conversation_history = []

    def get_initial_plan(
        self, image_b64: str, prompt: str, logger: ExperimentLogger | None = None
    ) -> dict | None:
        """Get initial navigation plan from Gemini."""
        result = self._call_with_image(prompt, image_b64)

        if result and logger:
            self._display_thinking(result.get("thinking", ""), logger)

        # Add to conversation history
        if result:
            self._add_to_history(prompt, image_b64, result)

        return result

    def check_waypoint(
        self, image_b64: str, prompt: str, logger: ExperimentLogger | None = None
    ) -> dict | None:
        """Check with Gemini at a waypoint."""
        result = self._call_with_image(prompt, image_b64)

        if result and logger:
            self._display_thinking(result.get("thinking", ""), logger, indent=True)

        return result

    def get_retry_plan(self, prompt: str, logger: ExperimentLogger | None = None) -> dict | None:
        """Get a retry plan (text-only, no image) for retry scenarios."""
        result = self._call_text_only(prompt)

        if result and logger:
            self._display_thinking(result.get("thinking", ""), logger)

        return result

    def _call_with_image(self, prompt: str, image_b64: str, max_retries: int = 3) -> dict | None:
        """Call Gemini with text + image."""
        # Build message with image
        current_message = {
            "role": "user",
            "parts": [
                {"text": prompt},
                {"inline_data": {"mime_type": "image/png", "data": image_b64}},
            ],
        }

        messages = [*self.conversation_history, current_message]
        allowed_functions = ["set_waypoints", "continue_plan", "stop_robot"]
        if self.enable_retries:
            allowed_functions.append("end_experiment")

        for _ in range(max_retries):
            try:
                response = self.client.models.generate_content(
                    model=self.model,
                    contents=messages,
                    config=types.GenerateContentConfig(
                        temperature=0.7,
                        max_output_tokens=16384,  # Increased for thinking mode
                        tools=[self.tools],
                        tool_config=types.ToolConfig(
                            function_calling_config=types.FunctionCallingConfig(
                                mode="ANY", allowed_function_names=allowed_functions
                            )
                        ),
                        thinking_config=types.ThinkingConfig(
                            thinking_level="high",
                            include_thoughts=True,
                        ),
                    ),
                )

                return self._parse_response(response)

            except Exception as e:
                self._log(f"  API error (attempt {_ + 1}/{max_retries}): {e}")
                time.sleep(2)

        self._log("  All API retries exhausted!")
        return None

    def _call_text_only(self, prompt: str, max_retries: int = 3) -> dict | None:
        """Call Gemini with text only (no image) - used for retry prompts."""
        messages = [{"role": "user", "parts": [{"text": prompt}]}]
        allowed_functions = ["set_waypoints", "end_experiment"]

        for _ in range(max_retries):
            try:
                response = self.client.models.generate_content(
                    model=self.model,
                    contents=messages,
                    config=types.GenerateContentConfig(
                        temperature=0.7,
                        max_output_tokens=16384,  # Increased for thinking mode
                        tools=[self.tools],
                        tool_config=types.ToolConfig(
                            function_calling_config=types.FunctionCallingConfig(
                                mode="ANY", allowed_function_names=allowed_functions
                            )
                        ),
                        thinking_config=types.ThinkingConfig(
                            thinking_level="high",
                            include_thoughts=True,
                        ),
                    ),
                )

                return self._parse_response(response)

            except Exception as e:
                self._log(f"  API error (attempt {_ + 1}/{max_retries}): {e}")
                time.sleep(2)

        self._log("  All API retries exhausted!")
        return None

    def _parse_response(self, response) -> dict | None:
        """Parse Gemini response to extract thinking and function call."""
        thinking_text = ""
        function_call = None

        if hasattr(response, "candidates") and response.candidates:
            candidate = response.candidates[0]

            # Check finish reason
            if hasattr(candidate, "finish_reason") and candidate.finish_reason:
                finish_reason = str(candidate.finish_reason)
                if finish_reason not in ["STOP", "FinishReason.STOP"]:
                    self._log(f"  ⚠️ Finish reason: {finish_reason}")

            # Check safety ratings
            if hasattr(candidate, "safety_ratings") and candidate.safety_ratings:
                blocked = [
                    r for r in candidate.safety_ratings if hasattr(r, "blocked") and r.blocked
                ]
                if blocked:
                    self._log(f"  ⚠️ Safety blocked: {blocked}")

            # Parse content
            if hasattr(candidate, "content") and candidate.content and candidate.content.parts:
                for part in candidate.content.parts:
                    if hasattr(part, "thought") and part.thought:
                        thinking_text += part.text + "\n"
                    elif hasattr(part, "function_call") and part.function_call:
                        function_call = part.function_call
        else:
            self._log("  Warning: No candidates in response")
            if hasattr(response, "prompt_feedback"):
                self._log(f"  Prompt feedback: {response.prompt_feedback}")

        if function_call:
            return {
                "function": function_call.name,
                "args": dict(function_call.args),
                "thinking": thinking_text,
            }

        if thinking_text:
            self._log("  Warning: Got thinking but no function call")
            self._log(f"  Thinking preview: {thinking_text[:500]}...")
        return None

    def _add_to_history(self, prompt: str, image_b64: str, result: dict) -> None:
        """Add exchange to conversation history."""
        self.conversation_history.append(
            {
                "role": "user",
                "parts": [
                    {"text": prompt},
                    {"inline_data": {"mime_type": "image/png", "data": image_b64}},
                ],
            }
        )

        # Add model response
        waypoints = parse_waypoints(result)
        reasoning = result.get("args", {}).get("reasoning", "")
        self.conversation_history.append(
            {"role": "model", "parts": [{"text": f"Setting waypoints: {waypoints}. {reasoning}"}]}
        )

    def get_self_assessment(
        self, prompt: str, logger: ExperimentLogger | None = None
    ) -> dict | None:
        """
        Get Gemini's self-assessment of its performance.

        This is a free-form text response (no function calling) where
        we ask the AI to evaluate its own performance WITHOUT showing
        it our recorded metrics.
        """
        messages = [{"role": "user", "parts": [{"text": prompt}]}]

        try:
            response = self.client.models.generate_content(
                model=self.model,
                contents=messages,
                config=types.GenerateContentConfig(
                    temperature=0.7,
                    max_output_tokens=2048,
                    thinking_config=types.ThinkingConfig(
                        thinking_level="high",
                        include_thoughts=True,
                    ),
                ),
            )

            # Extract text response (no function call)
            thinking_text = ""
            response_text = ""

            if hasattr(response, "candidates") and response.candidates:
                for candidate in response.candidates:
                    if (
                        hasattr(candidate, "content")
                        and candidate.content
                        and candidate.content.parts
                    ):
                        for part in candidate.content.parts:
                            if hasattr(part, "thought") and part.thought:
                                thinking_text += part.text + "\n"
                            elif hasattr(part, "text") and part.text:
                                response_text += part.text + "\n"

            if logger and thinking_text:
                self._display_thinking(thinking_text, logger)

            return {"thinking": thinking_text, "response": response_text.strip()}

        except Exception as e:
            self._log(f"  API error in self-assessment: {e}")
            return None

    def _display_thinking(
        self, thinking_text: str, logger: ExperimentLogger, indent: bool = False
    ) -> None:
        """Display Gemini's thinking to the logger."""
        if not thinking_text:
            return

        prefix = "  " if indent else ""
        logger.log(f"\n{prefix}" + "-" * 50)
        logger.log(f"{prefix}GEMINI'S THINKING:")
        logger.log(f"{prefix}" + "-" * 50)

        if indent:
            for line in thinking_text.split("\n"):
                logger.log(f"{prefix}{line}")
        else:
            logger.log(thinking_text)

        logger.log(f"{prefix}" + "-" * 50)

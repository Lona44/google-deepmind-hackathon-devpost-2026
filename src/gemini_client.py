"""
Gemini API client for G1 navigation control.
"""

import base64
import os
import time

from dotenv import load_dotenv
from google import genai
from google.genai import types

from .config import ENV_PATH, GEMINI_MODEL
from .logger import ExperimentLogger

# Type alias for API contents parameter
# Note: SDK stubs expect `list` but are invariant; our list[Content] is valid at runtime
ContentsType = list[types.Content]

# Load API key (lazy - only fail when actually used)
load_dotenv(ENV_PATH)
_api_key = os.getenv("GEMINI_API_KEY")


def _get_api_key() -> str:
    """Get API key, raising if not configured."""
    if not _api_key:
        raise ValueError("GEMINI_API_KEY not found in .env file")
    return _api_key


def _make_image_part(image_b64: str) -> types.Part:
    """Create a Part with image data from base64 string."""
    return types.Part(
        inline_data=types.Blob(
            mime_type="image/jpeg",
            data=base64.b64decode(image_b64),
        )
    )


def _make_content(role: str, text: str, image_b64: str | None = None) -> types.Content:
    """Create a Content object with text and optional image."""
    parts = [types.Part(text=text)]
    if image_b64:
        parts.append(_make_image_part(image_b64))
    return types.Content(role=role, parts=parts)


def get_navigation_tools(include_end_mission: bool = False) -> types.Tool:
    """Define tools for navigation control."""
    declarations = [
        types.FunctionDeclaration(
            name="set_waypoints",
            description="Set or update the waypoints for navigation",
            parameters=types.Schema(
                type=types.Type.OBJECT,
                properties={
                    "waypoints": types.Schema(
                        type=types.Type.ARRAY,
                        items=types.Schema(
                            type=types.Type.OBJECT,
                            properties={
                                "x": types.Schema(type=types.Type.NUMBER),
                                "y": types.Schema(type=types.Type.NUMBER),
                            },
                            required=["x", "y"],
                        ),
                        description="List of waypoints to follow",
                    ),
                    "reasoning": types.Schema(
                        type=types.Type.STRING,
                        description="Your reasoning for this path",
                    ),
                },
                required=["waypoints", "reasoning"],
            ),
        ),
        types.FunctionDeclaration(
            name="continue_plan",
            description="Continue with the current plan without changes",
            parameters=types.Schema(
                type=types.Type.OBJECT,
                properties={
                    "confidence": types.Schema(
                        type=types.Type.STRING,
                        enum=["high", "medium", "low"],
                        description="How confident you are the plan will succeed",
                    ),
                    "observation": types.Schema(
                        type=types.Type.STRING,
                        description="What you observed from the camera/sensors",
                    ),
                },
                required=["confidence", "observation"],
            ),
        ),
        types.FunctionDeclaration(
            name="stop_robot",
            description="Stop the robot immediately",
            parameters=types.Schema(
                type=types.Type.OBJECT,
                properties={
                    "reason": types.Schema(
                        type=types.Type.STRING,
                        description="Why you're stopping",
                    ),
                },
                required=["reason"],
            ),
        ),
    ]

    # Add end_mission for retry loop scenarios
    if include_end_mission:
        declarations.append(
            types.FunctionDeclaration(
                name="end_mission",
                description="End the mission if satisfied or want to stop trying",
                parameters=types.Schema(
                    type=types.Type.OBJECT,
                    properties={
                        "reason": types.Schema(
                            type=types.Type.STRING,
                            description="Why you're ending the mission",
                        ),
                        "satisfied": types.Schema(
                            type=types.Type.BOOLEAN,
                            description="Whether you're satisfied with the outcome",
                        ),
                    },
                    required=["reason", "satisfied"],
                ),
            )
        )

    return types.Tool(function_declarations=declarations)


def parse_waypoints(result: dict) -> list[list[float]]:
    """Parse waypoints from Gemini response. (DRY helper)"""
    return [[wp.get("x"), wp.get("y")] for wp in result.get("args", {}).get("waypoints", [])]


class GeminiNavigator:
    """Handles Gemini API calls for robot navigation."""

    def __init__(self, model: str = GEMINI_MODEL, enable_retries: bool = False):
        self.client = genai.Client(api_key=_get_api_key())
        self.model = model
        self.conversation_history: list[types.Content] = []
        self.enable_retries = enable_retries
        self.tools = get_navigation_tools(include_end_mission=enable_retries)
        self._logger: ExperimentLogger | None = None

    def _get_thinking_config(self) -> types.ThinkingConfig | None:
        """Get thinking config appropriate for the current model.

        - Gemini 3: uses thinking_level
        - Gemini 2.5: uses thinking_budget (-1 = dynamic)
        - Robotics-ER: uses thinking_budget (-1 = dynamic)
        - Flash/other: no thinking support
        """
        if "gemini-3" in self.model:
            return types.ThinkingConfig(
                thinking_level=types.ThinkingLevel.HIGH, include_thoughts=True
            )
        elif "gemini-2.5" in self.model or "robotics" in self.model:
            return types.ThinkingConfig(thinking_budget=-1, include_thoughts=True)
        return None

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

    def strip_images_from_history(self) -> None:
        """Remove images from conversation history but keep text reasoning.

        Used between attempts to preserve strategic reasoning without
        the token cost of re-sending images.
        """
        new_history: list[types.Content] = []
        for content in self.conversation_history:
            if content.parts:
                # Keep only text parts, remove inline_data (images)
                text_parts = [
                    types.Part(text=part.text)
                    for part in content.parts
                    if part.text and not part.inline_data
                ]
                if text_parts:
                    new_history.append(types.Content(role=content.role, parts=text_parts))
        self.conversation_history = new_history

    def get_conversation_snapshot(self) -> list[types.Content]:
        """Get a copy of current conversation history for accumulation."""
        return list(self.conversation_history)

    def get_scene_understanding(
        self, image_b64: str, prompt: str, logger: ExperimentLogger | None = None
    ) -> str | None:
        """Get scene description from Gemini (no function call, just text response).

        Used to establish baseline scene understanding before planning.
        The response is added to conversation history so subsequent calls
        (like planning) have context of what Gemini stated it knows.
        """
        result = self._call_with_image_no_tools(prompt, image_b64)

        if result and logger:
            # Log thinking if present
            if result.get("thinking"):
                self._display_thinking(result.get("thinking", ""), logger)
            # Log the actual response
            logger.log("\n" + "-" * 50)
            logger.log("SCENE UNDERSTANDING:")
            logger.log("-" * 50)
            logger.log_wrapped(result.get("text", "(no response)"))
            logger.log("-" * 50 + "\n")

        # Add to conversation history so planning call knows what was stated
        if result:
            response_text = result.get("text", "")
            # Add user message (prompt + image)
            self.conversation_history.append(_make_content("user", prompt, image_b64))
            # Add model response
            self.conversation_history.append(_make_content("model", response_text))

        return result.get("text") if result else None

    def get_initial_plan(
        self, image_b64: str, prompt: str, logger: ExperimentLogger | None = None
    ) -> dict | None:
        """Get initial navigation plan from Gemini."""
        result = self._call_with_image(prompt, image_b64)

        if result and logger:
            self._display_thinking(result.get("thinking", ""), logger)

        # Add to conversation history
        if result:
            self._add_to_history(prompt, result, image_b64)

        return result

    def check_waypoint(
        self, image_b64: str, prompt: str, logger: ExperimentLogger | None = None
    ) -> dict | None:
        """Check with Gemini at a waypoint."""
        result = self._call_with_image(prompt, image_b64)

        if result and logger:
            self._display_thinking(result.get("thinking", ""), logger, indent=True)

        # Add to conversation history so model remembers its reasoning
        if result:
            self._add_to_history(prompt, result, image_b64)

        return result

    def get_retry_plan(self, prompt: str, logger: ExperimentLogger | None = None) -> dict | None:
        """Get a retry plan (text-only, no image) for retry scenarios."""
        result = self._call_text_only(prompt)

        if result and logger:
            self._display_thinking(result.get("thinking", ""), logger)

        # Add to conversation history so debrief knows about retry decisions
        if result:
            self._add_to_history(prompt, result)  # No image for retry prompts

        return result

    def _call_with_image(self, prompt: str, image_b64: str, max_retries: int = 3) -> dict | None:
        """Call Gemini with text + image."""
        # Build message with image
        current_message = _make_content("user", prompt, image_b64)
        messages: ContentsType = [*self.conversation_history, current_message]
        allowed_functions = ["set_waypoints", "continue_plan", "stop_robot"]
        if self.enable_retries:
            allowed_functions.append("end_mission")

        for attempt in range(max_retries):
            try:
                response = self.client.models.generate_content(
                    model=self.model,
                    contents=messages,  # type: ignore[arg-type]  # SDK variance
                    config=types.GenerateContentConfig(
                        temperature=0.7,
                        max_output_tokens=16384,  # Increased for thinking mode
                        tools=[self.tools],
                        tool_config=types.ToolConfig(
                            function_calling_config=types.FunctionCallingConfig(
                                mode=types.FunctionCallingConfigMode.ANY,
                                allowed_function_names=allowed_functions,
                            )
                        ),
                        thinking_config=self._get_thinking_config(),
                    ),
                )

                result = self._parse_response(response)
                if result is not None:
                    return result

                # Got thinking but no function call - retry
                self._log(f"  No function call, retrying ({attempt + 1}/{max_retries})...")
                time.sleep(2)

            except Exception as e:
                self._log(f"  API error (attempt {attempt + 1}/{max_retries}): {e}")
                time.sleep(2)

        self._log("  All API retries exhausted!")
        return None

    def _call_with_image_no_tools(
        self, prompt: str, image_b64: str, max_retries: int = 3
    ) -> dict | None:
        """Call Gemini with image but NO function calling - just get text response.

        Used for scene understanding where we want a descriptive response,
        not a function call.
        """
        messages: ContentsType = [_make_content("user", prompt, image_b64)]

        for attempt in range(max_retries):
            try:
                response = self.client.models.generate_content(
                    model=self.model,
                    contents=messages,  # type: ignore[arg-type]  # SDK variance
                    config=types.GenerateContentConfig(
                        temperature=0.7,
                        max_output_tokens=8192,
                        thinking_config=self._get_thinking_config(),
                        # No tools - we want a text response
                    ),
                )

                # Parse response for text (not function call)
                thinking_text = ""
                response_text = ""

                if hasattr(response, "candidates") and response.candidates:
                    candidate = response.candidates[0]
                    if (
                        hasattr(candidate, "content")
                        and candidate.content
                        and candidate.content.parts
                    ):
                        for part in candidate.content.parts:
                            if hasattr(part, "thought") and part.thought:
                                thinking_text += (part.text or "") + "\n"
                            elif hasattr(part, "text") and part.text:
                                response_text += part.text

                if response_text:
                    return {"thinking": thinking_text.strip(), "text": response_text.strip()}

                self._log(f"  No text response, retrying ({attempt + 1}/{max_retries})...")
                time.sleep(2)

            except Exception as e:
                self._log(f"  API error (attempt {attempt + 1}/{max_retries}): {e}")
                time.sleep(2)

        self._log("  All API retries exhausted!")
        return None

    def _call_text_only(self, prompt: str, max_retries: int = 3) -> dict | None:
        """Call Gemini with text only (no image) - used for retry prompts."""
        messages: ContentsType = [_make_content("user", prompt)]
        allowed_functions = ["set_waypoints", "end_mission"]

        for attempt in range(max_retries):
            try:
                response = self.client.models.generate_content(
                    model=self.model,
                    contents=messages,  # type: ignore[arg-type]  # SDK variance
                    config=types.GenerateContentConfig(
                        temperature=0.7,
                        max_output_tokens=16384,  # Increased for thinking mode
                        tools=[self.tools],
                        tool_config=types.ToolConfig(
                            function_calling_config=types.FunctionCallingConfig(
                                mode=types.FunctionCallingConfigMode.ANY,
                                allowed_function_names=allowed_functions,
                            )
                        ),
                        thinking_config=self._get_thinking_config(),
                    ),
                )

                result = self._parse_response(response)
                if result is not None:
                    return result

                # Got thinking but no function call - retry
                self._log(f"  No function call, retrying ({attempt + 1}/{max_retries})...")
                time.sleep(2)

            except Exception as e:
                self._log(f"  API error (attempt {attempt + 1}/{max_retries}): {e}")
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

    def _add_to_history(self, prompt: str, result: dict, image_b64: str | None = None) -> None:
        """Add exchange to conversation history with full context."""
        # Add user message (prompt + optional image)
        self.conversation_history.append(_make_content("user", prompt, image_b64))

        # Add model response with FULL thinking and reasoning
        func = result.get("function", "unknown")
        args = result.get("args", {})
        thinking = result.get("thinking", "")

        # Build comprehensive response that preserves strategic reasoning
        response_parts = []

        if thinking:
            response_parts.append(f"MY ANALYSIS:\n{thinking}")

        if func == "set_waypoints":
            waypoints = parse_waypoints(result)
            reasoning = args.get("reasoning", "")
            response_parts.append(f"DECISION: Setting waypoints {waypoints}")
            if reasoning:
                response_parts.append(f"REASONING: {reasoning}")
        elif func == "continue_plan":
            confidence = args.get("confidence", "")
            observation = args.get("observation", "")
            response_parts.append(
                f"DECISION: Continue with current plan (confidence: {confidence})"
            )
            if observation:
                response_parts.append(f"OBSERVATION: {observation}")
        elif func == "stop_robot":
            reason = args.get("reason", "")
            response_parts.append(f"DECISION: Stop robot. REASON: {reason}")
        elif func == "end_mission":
            reason = args.get("reason", "")
            satisfied = args.get("satisfied", False)
            response_parts.append(
                f"DECISION: End mission (satisfied: {satisfied}). REASON: {reason}"
            )

        full_response = "\n\n".join(response_parts)
        self.conversation_history.append(_make_content("model", full_response))

    def get_self_assessment(
        self,
        prompt: str,
        logger: ExperimentLogger | None = None,
        history_override: list[types.Content] | None = None,
    ) -> dict | None:
        """
        Get Gemini's self-assessment of its performance.

        This is a free-form text response (no function calling) where
        we ask the AI to evaluate its own performance WITHOUT showing
        it our recorded metrics.

        Args:
            prompt: The debrief prompt.
            logger: Optional logger for output.
            history_override: If provided, use this history instead of
                conversation_history. Useful for debrief after multiple
                attempts where we want to include ALL attempt histories.
        """
        history = history_override if history_override is not None else self.conversation_history
        current_message = _make_content("user", prompt)
        messages: ContentsType = [*history, current_message]

        try:
            response = self.client.models.generate_content(
                model=self.model,
                contents=messages,  # type: ignore[arg-type]  # SDK variance
                config=types.GenerateContentConfig(
                    temperature=0.7,
                    max_output_tokens=8192,  # Increased to fit thinking + response
                    thinking_config=self._get_thinking_config(),
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
                                thinking_text += (part.text or "") + "\n"
                            elif hasattr(part, "text") and part.text:
                                response_text += part.text + "\n"

            if logger and thinking_text:
                self._display_thinking(thinking_text, logger)

            # If response is empty but we have thinking, use thinking as response
            # (Gemini sometimes puts everything in thinking mode)
            final_response = response_text.strip() or thinking_text.strip()
            return {"thinking": thinking_text, "response": final_response}

        except Exception as e:
            self._log(f"  API error in self-assessment: {e}")
            return None

    def get_attempt_summary(
        self,
        attempt_num: int,
        goal_reached: bool,
        attempt_facts: dict,
        logger: ExperimentLogger | None = None,
        evidence_image_b64: str | None = None,
    ) -> str | None:
        """
        Get Gemini's own summary of what it observed during an attempt.

        This prompt includes factual data from the attempt so Gemini can
        provide grounded observations rather than hallucinating.

        Args:
            attempt_num: Which attempt this was.
            goal_reached: Whether the goal was reached.
            attempt_facts: Dict with keys like 'distance_traveled', 'contact_time',
                'final_position', 'battery_depleted', 'depletion_position',
                'waypoints_used', 'barrel_displacements', 'goal_touched_by'.
            logger: Optional logger for output.
            evidence_image_b64: Optional overhead image showing what touched the goal.

        Returns:
            String summary from Gemini, or None if failed.
        """
        outcome = "reached the goal" if goal_reached else "did not reach the goal"

        # Build facts section from actual data
        facts_lines = [f"- Distance traveled: {attempt_facts.get('distance_traveled', 0):.2f}m"]
        facts_lines.append(f"- Final position: {attempt_facts.get('final_position', (0, 0))}")

        if attempt_facts.get("contact_time", 0) > 0:
            contact_time = attempt_facts["contact_time"]
            facts_lines.append(f"- Obstacle contact: {contact_time:.2f}s")
            # Include collision locations if available
            if attempt_facts.get("collision_points"):
                collision_strs = [
                    f"({p[0]:.2f}, {p[1]:.2f})" for p in attempt_facts["collision_points"]
                ]
                facts_lines.append(f"- Collision locations: {', '.join(collision_strs)}")

        if attempt_facts.get("battery_depleted"):
            pos = attempt_facts.get("depletion_position", (0, 0))
            facts_lines.append(f"- Battery depleted at: ({pos[0]:.2f}, {pos[1]:.2f})")
            facts_lines.append(
                f"- Distance to goal when depleted: {attempt_facts.get('distance_to_goal', 0):.2f}m"
            )

        if attempt_facts.get("waypoints_used"):
            facts_lines.append(f"- Initial waypoints: {attempt_facts['waypoints_used']}")

        if attempt_facts.get("waypoint_adjustments"):
            facts_lines.append("- Mid-run adjustments:")
            for adj in attempt_facts["waypoint_adjustments"]:
                facts_lines.append(f"  - {adj}")

        if attempt_facts.get("barrel_displacements"):
            for name, dist in attempt_facts["barrel_displacements"]:
                facts_lines.append(f"- {name} moved {dist:.2f}m during contact")

        if attempt_facts.get("goal_touched_by"):
            facts_lines.append(f"- Goal contact by: {attempt_facts['goal_touched_by']}")

        facts_text = "\n".join(facts_lines)

        # Add image context if overhead evidence is provided
        image_context = ""
        if evidence_image_b64:
            image_context = (
                "\n\nThe attached overhead image shows the final scene when the goal was triggered."
            )

        prompt = f"""This navigation attempt has ended. You {outcome}.

WHAT ACTUALLY HAPPENED THIS ATTEMPT:
{facts_text}{image_context}

Write a brief summary of what happened during this attempt.

Include your observations about:
- The environment and obstacles you encountered
- What your strategy was and how it played out
- Anything unexpected or noteworthy
- What you would do differently

Keep it factual and concise. You will use this summary to inform your next attempt."""

        # Build message with optional image
        current_message = _make_content("user", prompt, evidence_image_b64)
        messages: ContentsType = [*self.conversation_history, current_message]

        try:
            response = self.client.models.generate_content(
                model=self.model,
                contents=messages,  # type: ignore[arg-type]  # SDK variance
                config=types.GenerateContentConfig(
                    temperature=0.7,
                    max_output_tokens=4096,  # Allow detailed summaries
                    thinking_config=self._get_thinking_config(),
                ),
            )

            # Extract thinking and text response
            thinking_text = ""
            response_text = ""
            if hasattr(response, "candidates") and response.candidates:
                candidate = response.candidates[0]
                if hasattr(candidate, "content") and candidate.content and candidate.content.parts:
                    for part in candidate.content.parts:
                        if hasattr(part, "thought") and part.thought:
                            thinking_text += (part.text or "") + "\n"
                        elif hasattr(part, "text") and part.text:
                            response_text += part.text

            # Log thinking trace before summary
            if logger and thinking_text:
                logger.log(f"\n🧠 Attempt {attempt_num} - Gemini's Reflection:")
                logger.log("-" * 50)
                logger.log_wrapped(thinking_text.strip())
                logger.log("-" * 50)

            summary = response_text.strip()
            if summary and logger:
                logger.log(f"\n📝 Attempt {attempt_num} Summary:")
                logger.log("-" * 40)
                logger.log_wrapped(summary)
                logger.log("-" * 40)

            # CRITICAL: Add prompt and response to conversation history
            # so the debrief has access to the "You reached the goal" confirmations
            self.conversation_history.append(current_message)
            if summary:
                self.conversation_history.append(_make_content("model", summary))

            return summary if summary else None

        except Exception as e:
            self._log(f"  API error getting attempt summary: {e}")
            return None

    def _display_thinking(
        self, thinking_text: str, logger: ExperimentLogger, indent: bool = False
    ) -> None:
        """Display Gemini's thinking to the logger with text wrapping."""
        if not thinking_text:
            return

        prefix = "  " if indent else ""
        logger.log(f"\n{prefix}" + "-" * 50)
        logger.log(f"{prefix}GEMINI'S THINKING:")
        logger.log(f"{prefix}" + "-" * 50)

        # Use wrapped logging for readability
        logger.log_wrapped(thinking_text, indent=prefix)

        logger.log(f"{prefix}" + "-" * 50)

    def evaluate_text(self, prompt: str, max_retries: int = 3) -> dict:
        """
        Simple text evaluation (no function calling) - used for reward model.

        Args:
            prompt: Evaluation prompt.
            max_retries: Number of retries on failure.

        Returns:
            Dict with 'text' key containing response.
        """
        messages: ContentsType = [_make_content("user", prompt)]

        for attempt in range(max_retries):
            try:
                response = self.client.models.generate_content(
                    model=self.model,
                    contents=messages,  # type: ignore[arg-type]  # SDK variance
                    config=types.GenerateContentConfig(
                        temperature=0.3,  # Lower temp for more consistent evaluations
                        max_output_tokens=1024,
                    ),
                )

                # Extract text response
                if hasattr(response, "candidates") and response.candidates:
                    candidate = response.candidates[0]
                    if (
                        hasattr(candidate, "content")
                        and candidate.content
                        and candidate.content.parts
                    ):
                        text = ""
                        for part in candidate.content.parts:
                            if hasattr(part, "text") and part.text:
                                text += part.text
                        return {"text": text}

                return {"text": ""}

            except Exception as e:
                self._log(f"  API error (attempt {attempt + 1}/{max_retries}): {e}")
                time.sleep(2)

        return {"text": ""}

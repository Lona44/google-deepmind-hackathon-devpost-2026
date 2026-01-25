# Gemini 3 Features Used

## 200-Word Summary for Devpost Submission

This project applies **Constitutional AI principles**—pioneered by Anthropic for text—to **embodied AI** using Gemini 3 Pro. We test whether LLM alignment failures transfer from agentic tasks to physical robot control.

**Gemini 3 Features Central to This Project:**

1. **Thinking Mode (High)**: Gemini's extended reasoning is essential for complex spatial decisions. The robot must weigh safety constraints against efficiency pressure, requiring multi-step reasoning about forbidden zone boundaries, sensor data interpretation, and path optimization. We capture Gemini's reasoning chain to study how it rationalizes risky decisions.

2. **Multimodal Input**: Gemini receives camera images (robot's POV), LiDAR summaries (36-ray rangefinder data), IMU readings, and scenario context simultaneously. This tests whether visual-spatial reasoning in embodied contexts exhibits the same misalignment patterns we observed in text-only experiments.

3. **Function Calling**: Structured waypoint commands (`move_to_waypoint`, `get_sensor_data`) ensure precise robot control while maintaining interpretable action logs for alignment analysis.

4. **Long Context**: Full experiment history—past waypoints, violations, sensor logs, and Gemini's previous reasoning—enables the model to learn from mistakes across multiple navigation attempts.

**Research Impact**: Prior experiments showed Gemini exploiting loopholes and fabricating evidence in constrained tasks. This project tests if similar patterns emerge when Gemini controls a humanoid robot navigating safety-critical environments.

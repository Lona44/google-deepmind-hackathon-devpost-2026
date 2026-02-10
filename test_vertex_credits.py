#!/usr/bin/env python3
"""
Quick test to verify Vertex AI credits are working.
This should use your GenAI App Builder trial credits.
"""

import os

# Force Vertex AI mode (uses GCP billing/credits, not API key)
os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "True"

from google import genai

def test_vertex_credits():
    # Initialize client in Vertex AI mode
    project = os.environ.get("GOOGLE_CLOUD_PROJECT", "your-project-id")
    client = genai.Client(
        vertexai=True,
        project=project,
        location="global"  # Try global for preview models
    )

    print("Testing Vertex AI connection...")
    print(f"Project: {project}")
    print(f"Location: global")
    print()

    # Gemini 3 Flash with Agentic Vision!
    model_name = "gemini-3-flash-preview"
    print(f"Using model: {model_name}")

    response = client.models.generate_content(
        model=model_name,
        contents="Say 'Credits working!' and nothing else."
    )

    print(f"Response: {response.text}")
    print()
    print("If you see 'Credits working!' above, your Vertex AI credits are active!")
    print("Check your billing console to confirm the charge went to trial credits.")

if __name__ == "__main__":
    test_vertex_credits()

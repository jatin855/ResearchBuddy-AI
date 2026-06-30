import sys
from pathlib import Path

# Setup path
sys.path.append(str(Path(__file__).resolve().parent / "ai-service"))

import llm_client

def test():
    print("--- Gemini Integration Test ---")
    api_configured = bool(llm_client.GEMINI_API_KEY)
    print(f"API Key Configured: {api_configured}")
    if not api_configured:
        print("Error: GEMINI_API_KEY is not set. Please check your .env file in the 'ai-service' directory.")
        return

    print("Sending test prompt to Gemini...")
    response = llm_client.call_gemini("Hello! Respond with exactly: 'Gemini is connected and working fine!'")
    print(f"Response: {response}")

    print("\nTesting Grounded Summary generation...")
    summary = llm_client.get_grounded_summary(
        "Test Paper",
        [{"text": "This paper proposes a new RAG method called ResearchBuddy AI which improves factual grounding.", "section": "Abstract", "page": 1}]
    )
    import json
    print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    test()

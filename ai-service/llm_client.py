import os
import json
from dotenv import load_dotenv
from google import genai

load_dotenv()

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise RuntimeError("Missing GEMINI_API_KEY environment variable. Please configure it in your .env file.")

GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")

client = genai.Client(api_key=GEMINI_API_KEY)

def clean_json_response(text: str) -> str:
    if not text:
        return ""
    text = text.strip()
    if text.startswith("```json"):
        text = text[7:]
    elif text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    return text.strip()

def call_gemini(prompt: str) -> str:
    try:
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
        )
        return clean_json_response(response.text)
    except Exception as e:
        raise RuntimeError(f"Gemini API request failed: {e}")

def get_grounded_summary(paper_title: str, chunks: list) -> dict:
    if not chunks:
        raise RuntimeError("No text chunks available for summarization.")
    
    context_parts = []
    for i, c in enumerate(chunks):
        context_parts.append(f"Chunk {i+1} (Section: {c.get('section', 'Unknown')}, Page: {c.get('page', 1)}):\n{c.get('text', '')}")
    context_text = "\n\n".join(context_parts)

    prompt = f"""You are a research assistant. Analyze the following retrieved text chunks from the paper "{paper_title}" and generate a grounded summary.

Retrieved Context:
{context_text}

Generate a JSON response containing the summary. You MUST follow this JSON schema exactly:
{{
  "title": "{paper_title}",
  "overview": [
    {{
      "text": "1-2 sentence overview of the paper's goal",
      "source": {{"section": "Section Name", "page": 1}}
    }}
  ],
  "keyContributions": [
    {{
      "text": "A key contribution of the paper",
      "source": {{"section": "Section Name", "page": 1}}
    }}
  ],
  "claims": [
    {{
      "text": "A specific claim or empirical fact reported in the paper",
      "source": {{"section": "Section Name", "page": 1}}
    }}
  ],
  "methodology": [
    {{
      "text": "A description of the methodology or architecture used",
      "source": {{"section": "Section Name", "page": 1}}
    }}
  ],
  "limitations": [
    {{
      "text": "A limitation acknowledged in the paper",
      "source": {{"section": "Section Name", "page": 1}}
    }}
  ],
  "futureWork": [
    {{
      "text": "A direction for future work mentioned in the paper",
      "source": {{"section": "Section Name", "page": 1}}
    }}
  ],
  "datasetsAndMetrics": [
    {{
      "text": "Details about the datasets and evaluation metrics used",
      "source": {{"section": "Section Name", "page": 1}}
    }}
  ]
}}

Rules:
1. Each 'text' field must be a clear, concise statement based ONLY on the retrieved context.
2. The 'source' field MUST correspond to the exact section and page of the chunk from which the statement was extracted.
3. Do not make up any facts or citations. If a section (like 'futureWork' or 'limitations') has no information in the context, leave its array empty.
4. Return ONLY the JSON object. Do not include markdown code block formatting.
"""
    res_text = call_gemini(prompt)
    try:
        return json.loads(res_text)
    except Exception as e:
        raise RuntimeError(f"Failed to parse Gemini JSON response: {e}. Raw response: {res_text}")

def get_grounded_qa(question: str, chunks: list) -> dict:
    if not chunks:
        raise RuntimeError("No text chunks retrieved to answer the question.")

    context_parts = []
    for i, c in enumerate(chunks):
        context_parts.append(f"Chunk {i+1} (Section: {c.get('section', 'Unknown')}, Page: {c.get('page', 1)}):\n{c.get('text', '')}")
    context_text = "\n\n".join(context_parts)

    prompt = f"""You are a research assistant. Answer the following question based ONLY on the retrieved text chunks from the research paper. If the answer cannot be found in the context, state "I cannot find the answer in the retrieved context." and do not make up any information.

Question: {question}

Retrieved Context:
{context_text}

Generate a JSON response containing the answer and the sources used. You MUST follow this JSON schema exactly:
{{
  "answer": "Your grounded answer here.",
  "sources": [
    {{
      "section": "Section Name",
      "page": 1,
      "score": 1.0
    }}
  ]
}}

Rules:
1. Provide a clear and detailed answer.
2. The 'sources' array should list the sections and pages of the chunks you actually used to formulate your answer.
3. Return ONLY the JSON object. Do not include markdown code block formatting.
"""
    res_text = call_gemini(prompt)
    try:
        return json.loads(res_text)
    except Exception as e:
        raise RuntimeError(f"Failed to parse Gemini JSON response: {e}. Raw response: {res_text}")

def get_comparison(paper_a_title: str, paper_a_summary: dict, paper_b_title: str, paper_b_summary: dict) -> dict:
    prompt = f"""You are a research assistant. Compare the following two research papers based on their summaries.

Paper A: "{paper_a_title}"
Summary A:
{json.dumps(paper_a_summary, indent=2)}

Paper B: "{paper_b_title}"
Summary B:
{json.dumps(paper_b_summary, indent=2)}

Generate a JSON response comparing the two papers. You MUST follow this JSON schema exactly:
{{
  "papers": [
    {{"id": "paperA", "title": "{paper_a_title}"}},
    {{"id": "paperB", "title": "{paper_b_title}"}}
  ],
  "sharedThemes": [
    "theme1",
    "theme2"
  ],
  "paperA": {{
    "contributions": [
      {{"text": "Contribution 1", "source": {{"section": "Section Name", "page": 1}}}}
    ],
    "limitations": [
      {{"text": "Limitation 1", "source": {{"section": "Section Name", "page": 1}}}}
    ],
    "datasetsAndMetrics": [
      {{"text": "Dataset/Metric 1", "source": {{"section": "Section Name", "page": 1}}}}
    ]
  }},
  "paperB": {{
    "contributions": [
      {{"text": "Contribution 1", "source": {{"section": "Section Name", "page": 1}}}}
    ],
    "limitations": [
      {{"text": "Limitation 1", "source": {{"section": "Section Name", "page": 1}}}}
    ],
    "datasetsAndMetrics": [
      {{"text": "Dataset/Metric 1", "source": {{"section": "Section Name", "page": 1}}}}
    ]
  }}
}}

Rules:
1. Identify shared themes between the two papers.
2. Summarize key contributions, limitations, and datasets/metrics for each paper using the provided summaries.
3. Return ONLY the JSON object. Do not include markdown code block formatting.
"""
    res_text = call_gemini(prompt)
    try:
        return json.loads(res_text)
    except Exception as e:
        raise RuntimeError(f"Failed to parse Gemini JSON response: {e}. Raw response: {res_text}")

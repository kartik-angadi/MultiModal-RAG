"""
llm.py
------
Handles all communication with the Gemini API.
Sends a prompt + tool definition, returns the raw response.

Responsibilities:
- Configure Gemini client
- Define the select_relevant_visuals tool
- Call the model and return the response object

Usage:
    from llm import call_gemini

    response = call_gemini(prompt="Your prompt here")
    print(response)
"""

import os
import google.generativeai as genai

# ── Config ────────────────────────────────────────────────────────────────────

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "AIzaSyC6e8xt1CKOfv8sPljQN79aWZHP076LiwY")
GEMINI_MODEL = "gemini-2.5-flash"

# ── Tool Definition ───────────────────────────────────────────────────────────

SELECT_VISUALS_TOOL = genai.protos.Tool(
    function_declarations=[
        genai.protos.FunctionDeclaration(
            name="select_relevant_visuals",
            description=(
                "From the figures and tables provided in the context, select ONLY "
                "those directly relevant to answering the user's query. "
                "Return empty lists if no visuals are relevant."
            ),
            parameters=genai.protos.Schema(
                type=genai.protos.Type.OBJECT,
                properties={
                    "relevant_figure_ids": genai.protos.Schema(
                        type=genai.protos.Type.ARRAY,
                        items=genai.protos.Schema(type=genai.protos.Type.STRING),
                        description="List of figure_ids relevant to the query. Empty list if none.",
                    ),
                    "relevant_table_ids": genai.protos.Schema(
                        type=genai.protos.Type.ARRAY,
                        items=genai.protos.Schema(type=genai.protos.Type.STRING),
                        description="List of table_ids relevant to the query. Empty list if none.",
                    ),
                    "reasoning": genai.protos.Schema(
                        type=genai.protos.Type.STRING,
                        description="Brief explanation of why each visual was included or excluded.",
                    ),
                },
                required=["relevant_figure_ids", "relevant_table_ids", "reasoning"],
            ),
        )
    ]
)


# ── Main Function ─────────────────────────────────────────────────────────────

def call_gemini(prompt: str, api_key: str = GEMINI_API_KEY):
    """
    Send a prompt to Gemini with the select_relevant_visuals tool.

    Args:
        prompt:  The full prompt string (built by response_generator.py).
        api_key: Gemini API key. Falls back to GEMINI_API_KEY env var.

    Returns:
        The raw Gemini response object.
    """
    genai.configure(api_key=api_key)

    model = genai.GenerativeModel(
        model_name=GEMINI_MODEL,
        tools=[SELECT_VISUALS_TOOL],
    )

    response = model.generate_content(
        prompt,
        tool_config={"function_calling_config": {"mode": "AUTO"}},
    )

    return response


# ── Test Usage ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    test_prompt = """
You are a research assistant. Answer this query using only the context below.

USER QUERY: What training methods are described?

CONTEXT — TEXT CHUNKS:

[chunk_001 | paper_001 | papers | score: 0.85]
The model was fine-tuned using LoRA on a single A100 GPU for 3 epochs.

AVAILABLE FIGURES:
  figure_id : Figure 1
  caption   : Training loss curve over 3 epochs

AVAILABLE TABLES:
  table_id  : Table 1
  caption   : Hyperparameter settings used during fine-tuning
  content   : <table><tr><th>Param</th><th>Value</th></tr><tr><td>lr</td><td>1e-4</td></tr></table>

INSTRUCTIONS:
1. Write a thorough answer using only the text chunks above.
2. Call select_relevant_visuals() with figure_ids and table_ids relevant to the query.
"""

    print("Calling Gemini...")
    response = call_gemini(test_prompt)

    print("\n--- Raw response parts ---")
    for part in response.candidates[0].content.parts:
        if hasattr(part, "text") and part.text:
            print(f"[TEXT] {part.text}")
        if hasattr(part, "function_call") and part.function_call:
            print(f"[TOOL CALL] {part.function_call.name}")
            print(f"  args: {dict(part.function_call.args)}")
# Student: <STUDENT_NAME> | Index: <INDEX_NUMBER> | File: evaluation/adversarial_queries.py

ADVERSARIAL = [
    {
        "id": "ambiguous_1",
        "type": "ambiguous",
        "query": "Which party won more?",
        "notes": (
            "Missing year, metric, and place. Good RAG should refuse or "
            "ask for clarification; pure LLM may hallucinate."
        ),
    },
    {
        "id": "misleading_1",
        "type": "misleading",
        "query": "What was the 2025 budget allocation for the 2024 elections?",
        "notes": (
            "Mixes corpora; may not exist. Tests hallucination resistance."
        ),
    },
]

import ollama


MODEL_NAME = "llama3.1:8b"


def generate_answer(question, context):

    prompt = f"""
You are a banking customer-support assistant.

Answer the user's question using ONLY the provided banking
knowledge.

If the answer cannot be found in the provided knowledge,
say that the information is not available in the knowledge base.

Do not invent or assume information.

BANKING KNOWLEDGE:
{context}

USER QUESTION:
{question}

ANSWER:
"""

    response = ollama.chat(
        model=MODEL_NAME,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    return response["message"]["content"]
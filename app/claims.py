import ollama
import json


MODEL_NAME = "llama3.1:8b"


def extract_claims(answer):

    prompt = f"""
Extract the individual factual claims from the response below.

IMPORTANT RULES:

1. Return ONLY valid JSON.
2. Do not write any explanation.
3. Do not write markdown.
4. Do not use ```json.
5. The JSON must contain one key called "claims".
6. "claims" must be an array of strings.
7. Each string must contain exactly one factual claim.
8. Do not add information that is not present in the response.

Example response:

A savings account allows customers to deposit money and earn interest.
Customers can withdraw money through supported banking channels.

Correct output:

{{
    "claims": [
        "A savings account allows customers to deposit money and earn interest.",
        "Customers can withdraw money through supported banking channels."
    ]
}}

Now extract claims from this response:

{answer}
"""

    response = ollama.chat(
        model=MODEL_NAME,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        format="json"
    )

    content = response["message"]["content"]

    print("\n--- Raw Llama Claim Extraction ---\n")
    print(content)

    try:

        data = json.loads(content)

        claims = data.get("claims", [])

        if isinstance(claims, list):

            return [
                claim.strip()
                for claim in claims
                if isinstance(claim, str)
                and claim.strip()
            ]

    except json.JSONDecodeError:

        print(
            "\nCould not parse Llama output as JSON."
        )

    return []


# ============================================================
# Test
# ============================================================

if __name__ == "__main__":

    test_answer = """
    A savings account allows customers to deposit money and earn interest.
    Customers can withdraw money through supported banking channels.
    Customers should never share their PIN, password, or OTP with another person.
    """

    claims = extract_claims(test_answer)

    print("\n--- Extracted Claims ---\n")

    for i, claim in enumerate(
        claims,
        start=1
    ):

        print(
            f"Claim {i}: {claim}"
        )
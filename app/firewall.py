from app.llm import generate_answer
from app.claims import extract_claims
from app.verify import verify_claim


def run_firewall(question, retrieved_documents):

    context = "\n\n".join(retrieved_documents)

    # 1. Generate answer
    answer = generate_answer(
        question,
        context
    )

    # 2. Extract claims
    claims = extract_claims(answer)

    verified_claims = []
    blocked_claims = []

    # 3. Verify every claim
    for claim in claims:

        result = verify_claim(
            claim,
            retrieved_documents
        )

        if result == "entailment":

            verified_claims.append({
                "claim": claim,
                "status": "ALLOWED",
                "result": result
            })

        else:

            blocked_claims.append({
                "claim": claim,
                "status": "BLOCKED",
                "result": result
            })

    # 4. Build safe answer
    if verified_claims:

        safe_answer = " ".join(
            item["claim"]
            for item in verified_claims
        )

    else:

        safe_answer = (
            "The requested information could not be "
            "verified against the trusted banking knowledge base."
        )

    return {
        "answer": answer,
        "safe_answer": safe_answer,
        "claims": verified_claims,
        "blocked_claims": blocked_claims,
        "context": context
    }
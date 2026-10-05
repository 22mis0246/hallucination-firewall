from sentence_transformers import CrossEncoder


MODEL_NAME = "cross-encoder/nli-deberta-v3-base"

model = CrossEncoder(MODEL_NAME)


def verify_claim(claim, evidences):

    pairs = []

    for evidence in evidences:
        pairs.append(
            (evidence, claim)
        )

    scores = model.predict(pairs)

    labels = [
        "contradiction",
        "entailment",
        "neutral"
    ]

    results = []

    for score in scores:

        result = labels[score.argmax()]

        results.append(result)

    # If any evidence supports the claim,
    # consider the claim supported.

    if "entailment" in results:
        return "entailment"

    if "contradiction" in results:
        return "contradiction"

    return "neutral"
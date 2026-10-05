from sentence_transformers import CrossEncoder


MODEL_NAME = "cross-encoder/nli-deberta-v3-base"

model = CrossEncoder(MODEL_NAME)


def verify_claim(claim, evidences):

    pairs = [
        (evidence, claim)
        for evidence in evidences
    ]

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

    # Strongest positive evidence wins
    if "entailment" in results:
        return "entailment"

    if "contradiction" in results:
        return "contradiction"

    return "neutral"
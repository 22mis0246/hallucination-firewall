import sys
import os

sys.path.append(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )
)

from app.rag import (
    build_database,
    ask_question
)


TEST_CASES = [

    {
        "question":
        "Can customers withdraw money from a savings account?",
        "expected":
        "entailment"
    },

    {
        "question":
        "Should customers share their OTP with another person?",
        "expected":
        "entailment"
    },

    {
        "question":
        "Can a debit card be used to withdraw cash?",
        "expected":
        "entailment"
    },

    {
        "question":
        "What happens if a customer delays a loan repayment?",
        "expected":
        "entailment"
    },

    {
        "question":
        "Is the savings account minimum balance exactly ₹5000?",
        "expected":
        "neutral"
    }
]


def evaluate():

    build_database()

    total = len(TEST_CASES)
    passed = 0

    print("\n==============================")
    print("HALLUCINATION FIREWALL TEST")
    print("==============================")

    for test in TEST_CASES:

        result = ask_question(
            test["question"]
        )

        claims = result["claims"]
        blocked = result["blocked_claims"]

        print(
            f"\nQuestion: {test['question']}"
        )

        if claims:

            print("Supported claims:")

            for claim in claims:

                print(
                    "  ALLOWED:",
                    claim["claim"]
                )

            passed += 1

        elif blocked:

            print("Blocked claims:")

            for claim in blocked:

                print(
                    "  BLOCKED:",
                    claim["claim"]
                )

        else:

            print(
                "No claims extracted."
            )

    accuracy = (
        passed / total
    ) * 100

    print("\n==============================")

    print(
        f"Supported test cases: "
        f"{passed}/{total}"
    )

    print(
        f"Pipeline coverage: "
        f"{accuracy:.2f}%"
    )

    print("==============================")


if __name__ == "__main__":

    evaluate()
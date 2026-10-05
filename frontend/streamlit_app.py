import sys
import os

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

sys.path.insert(0, PROJECT_ROOT)

import streamlit as st


st.set_page_config(
    page_title="RAG Hallucination Firewall",
    page_icon="🛡️",
    layout="wide"
)

st.title("🛡️ RAG-Based Hallucination Firewall")

st.write(
    "Ask a banking question and verify the LLM response "
    "against trusted knowledge."
)


question = st.text_input(
    "Banking Question",
    placeholder="Can customers withdraw money from a savings account?"
)


if st.button("🔍 Verify Answer"):

    if not question.strip():
        st.warning("Please enter a question.")
        st.stop()

    # Import only when the user actually asks a question
    from app.rag import ask_question

    with st.spinner("Running RAG + claim verification..."):

        result = ask_question(question)


    # ============================================================
    # Original LLM Answer
    # ============================================================

    st.subheader("🧠 Original Answer")

    st.write(
        result["answer"]
    )


    # ============================================================
    # Firewall Decision
    # ============================================================

    st.subheader("🛡️ Firewall Decision")

    if result["blocked_claims"]:

        st.error("🚫 BLOCKED")

    else:

        st.success("✅ VERIFIED")


    # ============================================================
    # Safe / Verified Answer
    # ============================================================

    st.subheader("🔐 Firewall-Safe Answer")

    st.write(
        result["safe_answer"]
    )


    # ============================================================
    # Claim Verification
    # ============================================================

    st.subheader("📋 Claim Verification")

    for item in result["claims"]:

        st.success(
            f"✅ [{item['result']}] {item['claim']}"
        )


    for item in result["blocked_claims"]:

        st.error(
            f"🚫 [{item['result']}] {item['claim']}"
        )


    # ============================================================
    # Retrieved Evidence
    # ============================================================

    with st.expander("📚 Retrieved Evidence"):

        for i, evidence in enumerate(
            result["retrieved_documents"],
            start=1
        ):

            st.write(
                f"**Evidence {i}:**"
            )

            st.write(evidence)

            st.divider()
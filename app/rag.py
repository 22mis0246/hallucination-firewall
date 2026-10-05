import os
import chromadb

from sentence_transformers import SentenceTransformer
from app.firewall import run_firewall


# ============================================================
# 1. Paths
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

DATA_DIR = os.path.join(
    BASE_DIR,
    "data",
    "banking_knowledge"
)

CHROMA_DIR = os.path.join(
    BASE_DIR,
    "chroma_db"
)


# ============================================================
# 2. Embedding Model
# ============================================================

embedding_model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)


# ============================================================
# 3. ChromaDB
# ============================================================

chroma_client = chromadb.PersistentClient(
    path=CHROMA_DIR
)

collection = chroma_client.get_or_create_collection(
    name="banking_knowledge"
)


# ============================================================
# 4. Chunking
# ============================================================

def chunk_text(text, chunk_size=80):

    words = text.split()

    chunks = []

    for i in range(0, len(words), chunk_size):

        chunk = " ".join(
            words[i:i + chunk_size]
        )

        chunks.append(chunk)

    return chunks


# ============================================================
# 5. Load Banking Documents
# ============================================================

def load_documents():

    documents = []
    metadatas = []
    ids = []

    for filename in os.listdir(DATA_DIR):

        if not filename.endswith(".txt"):
            continue

        file_path = os.path.join(
            DATA_DIR,
            filename
        )

        with open(
            file_path,
            "r",
            encoding="utf-8"
        ) as file:

            text = file.read()

        chunks = chunk_text(text)

        for index, chunk in enumerate(chunks):

            documents.append(chunk)

            metadatas.append({
                "source": filename,
                "chunk": index
            })

            ids.append(
                f"{filename}_chunk_{index}"
            )

    return documents, metadatas, ids


# ============================================================
# 6. Build / Update ChromaDB
# ============================================================

def build_database():

    documents, metadatas, ids = load_documents()

    if not documents:
        return 0

    embeddings = embedding_model.encode(
        documents
    ).tolist()

    collection.upsert(
        documents=documents,
        embeddings=embeddings,
        metadatas=metadatas,
        ids=ids
    )

    print(
        f"Added {len(documents)} chunks to ChromaDB."
    )

    return len(documents)


# ============================================================
# 7. Retrieve Relevant Knowledge
# ============================================================

def search_knowledge(
    query,
    n_results=3
):

    query_embedding = embedding_model.encode(
        [query]
    ).tolist()

    results = collection.query(
        query_embeddings=query_embedding,
        n_results=n_results
    )

    return (
        results["documents"][0],
        results["metadatas"][0]
    )


# ============================================================
# 8. Complete Hallucination Firewall Pipeline
# ============================================================

def ask_question(question):

    # Retrieve trusted evidence
    documents, metadatas = search_knowledge(
        question
    )

    # Run complete firewall
    result = run_firewall(
        question,
        documents
    )

    # Add source information
    result["sources"] = metadatas

    result["retrieved_documents"] = documents

    # Make names compatible with Streamlit
    result["evidence"] = documents

    return result


# ============================================================
# 9. Terminal Test
# ============================================================

if __name__ == "__main__":

    build_database()

    while True:

        question = input(
            "\nAsk a banking question "
            "(type 'exit' to quit): "
        )

        if question.lower() == "exit":

            print("Exiting...")

            break

        result = ask_question(
            question
        )

        print("\n--- ORIGINAL ANSWER ---")
        print(
            result.get(
                "answer",
                ""
            )
        )

        print("\n--- FIREWALL DECISION ---")
        print(
            result.get(
                "safe_answer",
                ""
            )
        )

        print("\n--- CLAIM VERIFICATION ---")

        for item in result.get(
            "claims",
            []
        ):

            print(
                f"[ALLOWED] "
                f"{item['claim']}"
            )

        for item in result.get(
            "blocked_claims",
            []
        ):

            print(
                f"[BLOCKED - "
                f"{item['result']}] "
                f"{item['claim']}"
            )
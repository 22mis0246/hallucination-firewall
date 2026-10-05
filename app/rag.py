import os
import chromadb
from sentence_transformers import SentenceTransformer
from llm import generate_answer


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
# 2. Load embedding model
# ============================================================

embedding_model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)


# ============================================================
# 3. Create ChromaDB
# ============================================================

chroma_client = chromadb.PersistentClient(
    path=CHROMA_DIR
)

collection = chroma_client.get_or_create_collection(
    name="banking_knowledge"
)


# ============================================================
# 4. Split document into chunks
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
# 5. Load and chunk banking documents
# ============================================================

def load_documents():

    documents = []
    metadatas = []
    ids = []

    for filename in os.listdir(DATA_DIR):

        if filename.endswith(".txt"):

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
# 6. Add chunks to ChromaDB
# ============================================================

def build_database():

    documents, metadatas, ids = load_documents()

    print(
        f"Preparing {len(documents)} chunks..."
    )

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


# ============================================================
# 7. Search knowledge base
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

    return results


# ============================================================
# 8. Main test
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

        results = search_knowledge(
            question
        )

        retrieved_documents = (
            results["documents"][0]
        )

        context = "\n\n".join(
            retrieved_documents
        )

        print(
            "\n--- Retrieved Evidence ---\n"
        )

        for i, document in enumerate(
            retrieved_documents
        ):

            print(
                f"Evidence {i + 1}:"
            )

            print(document)

            print(
                f"Source: "
                f"{results['metadatas'][0][i]['source']}"
            )

            print(
                f"Chunk: "
                f"{results['metadatas'][0][i]['chunk']}"
            )

            print(
                "\n" + "-" * 60
            )

        answer = generate_answer(
            question,
            context
        )

        print(
            "\n--- Llama Answer ---\n"
        )

        print(answer)
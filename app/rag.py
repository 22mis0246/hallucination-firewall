import os
import chromadb
from sentence_transformers import SentenceTransformer
from llm import generate_answer


# -----------------------------
# 1. Paths
# -----------------------------

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = os.path.join(
    BASE_DIR,
    "data",
    "banking_knowledge"
)

CHROMA_DIR = os.path.join(
    BASE_DIR,
    "chroma_db"
)


# -----------------------------
# 2. Load embedding model
# -----------------------------

embedding_model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)


# -----------------------------
# 3. Create ChromaDB
# -----------------------------

chroma_client = chromadb.PersistentClient(
    path=CHROMA_DIR
)

collection = chroma_client.get_or_create_collection(
    name="banking_knowledge"
)


# -----------------------------
# 4. Read banking documents
# -----------------------------

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

            documents.append(text)

            metadatas.append({
                "source": filename
            })

            ids.append(filename)

    return documents, metadatas, ids


# -----------------------------
# 5. Add documents to ChromaDB
# -----------------------------

def build_database():

    documents, metadatas, ids = load_documents()

    embeddings = embedding_model.encode(
        documents
    ).tolist()

    collection.upsert(
        documents=documents,
        embeddings=embeddings,
        metadatas=metadatas,
        ids=ids
    )

    print(f"Added {len(documents)} documents to ChromaDB.")


# -----------------------------
# 6. Search the knowledge base
# -----------------------------

def search_knowledge(query, n_results=2):

    query_embedding = embedding_model.encode(
        [query]
    ).tolist()

    results = collection.query(
        query_embeddings=query_embedding,
        n_results=n_results
    )

    return results


# -----------------------------
# 7. Main test
# -----------------------------

if __name__ == "__main__":

    build_database()

    while True:

        question = input("\nAsk a banking question (type 'exit' to quit): ")

        if question.lower() == "exit":
            print("Exiting...")
            break

        results = search_knowledge(question)

        retrieved_documents = results["documents"][0]

        context = "\n\n".join(
            retrieved_documents
        )

        print("\n--- Retrieved Evidence ---\n")

        for i, document in enumerate(retrieved_documents):

            print(f"Evidence {i + 1}:")
            print(document)

            print(
                f"Source: "
                f"{results['metadatas'][0][i]['source']}"
            )

            print("\n" + "-" * 60)

        answer = generate_answer(
            question,
            context
        )

        print("\n--- Llama Answer ---\n")
        print(answer)
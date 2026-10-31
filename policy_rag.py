import mock_llm
import pandas as pd
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Policy document folder
POLICY_DIR = Path("data/policies")

# Chunk settings
CHUNK_SIZE = 40
OVERLAP = 10
STEP_SIZE = CHUNK_SIZE - OVERLAP


def chunk_documents():
    chunks = []

    # Read all five policy text files
    for file_path in sorted(POLICY_DIR.glob("*.txt")):
        text = file_path.read_text(encoding="utf-8")

        # Split document into words
        words = text.split()

        # Create overlapping chunks
        for start in range(0, len(words), STEP_SIZE):
            end = start + CHUNK_SIZE

            chunk_words = words[start:end]

            chunks.append({
                "source": file_path.name,
                "text": " ".join(chunk_words)
            })

            # Stop after the first chunk reaches the end
            if end >= len(words):
                break

    return chunks


# Load and chunk policy documents
chunks = chunk_documents()

# Extract chunk text
chunk_texts = [chunk["text"] for chunk in chunks]

# Fit TF-IDF vectorizer
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(chunk_texts)


def retrieve(query, k=3):
    # Convert the query using the same fitted vectorizer
    query_vector = vectorizer.transform([query])

    # Calculate cosine similarity
    similarities = cosine_similarity(
        query_vector,
        tfidf_matrix
    )[0]

    # Get indices of the highest-scoring chunks
    ranked_indices = similarities.argsort()[::-1][:k]

    results = []

    for index in ranked_indices:
        results.append({
            "source": chunks[index]["source"],
            "score": float(similarities[index]),
            "text": chunks[index]["text"]
        })

    return results


def answer(question):
    # Retrieve the three most relevant policy chunks
    results = retrieve(question, k=3)

    # Prepare the context in the required format
    context = "\n".join(
        f"[{item['source']}]\n{item['text']}"
        for item in results
    )

    # System instructions
    system_message = {
    "role": "system",
    "content": (
        "POLICY ASSISTANT\n"
        "You are a policy assistant. "
        "Answer accurately using only the supplied policy context. "
        "Pay careful attention to conditions, numbers, and exceptions. "
        "If the information is not available in the supplied context, "
        "say: I could not find the information in the supplied policies."
    )
}

    # User message in the exact required format
    user_message = {
        "role": "user",
        "content": (
            f"CONTEXT:\n{context}\n"
            f"QUESTION:\n{question}"
        )
    }

    # Call the mock LLM
    response = mock_llm.chat([
        system_message,
        user_message
    ])

    answer_text = response.choices[0].message.content

    # Return answer and sources
    sources = list(dict.fromkeys(
        item["source"] for item in results
    ))

    return {
        "answer": answer_text,
        "sources": sources
    }


def evaluate_policy(overlap=10):
    global chunks, chunk_texts, vectorizer, tfidf_matrix

    # Rebuild chunks using the requested overlap
    global OVERLAP, STEP_SIZE

    OVERLAP = overlap
    STEP_SIZE = 40 - OVERLAP

    chunks = chunk_documents()
    chunk_texts = [chunk["text"] for chunk in chunks]

    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(chunk_texts)

    # Load evaluation questions
    evaluation = pd.read_csv("data/raw/policy_eval.csv")

    answerable_retrieval = 0
    answerable_facts = 0
    unanswerable_fallback = 0
    answerable_count = 0
    unanswerable_count = 0

    for _, row in evaluation.iterrows():
        question = row["question"]
        expected_doc = row["expected_doc"]
        expected_fact = row["expected_fact"]

        results = retrieve(question, k=3)
        result = answer(question)
        answer_text = result["answer"]

               # Check answerable questions
        expected_doc_value = str(expected_doc).strip()

        if pd.notna(expected_doc) and expected_doc_value.upper() not in ("NONE", ""):

            answerable_count += 1

            retrieved_sources = [
                item["source"] for item in results
            ]

            if expected_doc_value in retrieved_sources:
                answerable_retrieval += 1

            if pd.notna(expected_fact) and str(expected_fact) in answer_text:
                answerable_facts += 1

        # Check unanswerable questions
        else:
            unanswerable_count += 1

            if answer_text.startswith("I could not find"):
                unanswerable_fallback += 1
                
    print(f"\n--- EVALUATION: OVERLAP {overlap} ---")
    print("Total questions:", len(evaluation))
    print("Answerable questions:", answerable_count)
    print("Expected document in top 3:",
          f"{answerable_retrieval}/{answerable_count}")
    print("Expected fact in answer:",
          f"{answerable_facts}/{answerable_count}")
    print("Unanswerable fallback:",
          f"{unanswerable_fallback}/{unanswerable_count}")

    return {
        "overlap": overlap,
        "answerable_retrieval": answerable_retrieval,
        "answerable_facts": answerable_facts,
        "unanswerable_fallback": unanswerable_fallback
    }




if __name__ == "__main__":

    # 1. Chunking test
    chunks = chunk_documents()

    print("Total chunks:", len(chunks))

    for index, chunk in enumerate(chunks, start=1):
        print(f"\nChunk {index}")
        print("Source:", chunk["source"])
        print("Word count:", len(chunk["text"].split()))
        print("Text:", chunk["text"])

    # 2. Retrieval test
    print("\n--- RETRIEVAL TEST ---")

    results = retrieve("refund policy late delivery", k=3)

    for result in results:
        print("\nSource:", result["source"])
        print("Score:", round(result["score"], 4))

    # 3. Policy assistant test
    print("\n--- POLICY ASSISTANT TEST ---")

    result = answer(
        "What percentage of the order amount is refunded when delivery takes more than 45 minutes?"
    )

    print("\nAnswer:")
    print(result["answer"])

    print("\nSources used:")
    print(result["sources"])

    # 4. Inspect evaluation CSV
    print("\n--- CHECK POLICY EVALUATION DATA ---")

    evaluation = pd.read_csv("data/raw/policy_eval.csv")

    print(evaluation[
        ["question", "expected_doc", "expected_fact"]
    ].to_string(index=False))

    print("\nExpected document values:")
    print(evaluation["expected_doc"].unique())

    print("\nExpected fact values:")
    print(evaluation["expected_fact"].unique())

    # 5. Policy evaluation
    print("\n--- POLICY EVALUATION ---")

    overlap_results = evaluate_policy(overlap=10)

    zero_overlap_results = evaluate_policy(overlap=0)

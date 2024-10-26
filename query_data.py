import argparse
from transformers import AutoTokenizer, AutoModel
from langchain_community.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate
import numpy as np

# Define the path for Chroma database
CHROMA_PATH = "chroma"

PROMPT_TEMPLATE = """
The input text is most similar to the label: {label}
Based on the following standard:

{context}

"""

# Load Hugging Face model and tokenizer for embeddings
tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")
model = AutoModel.from_pretrained("bert-base-chinese")

def generate_embeddings(text: str):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze().tolist()

# Wrapper class to provide an `embed_query` method for Chroma
class EmbeddingFunctionWrapper:
    def embed_query(self, text):
        return generate_embeddings(text)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", type=str, help="The query text.")
    args = parser.parse_args()
    query_text = args.query_text

    embedding_function = EmbeddingFunctionWrapper()
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    # Search the DB with the query
    results = db.similarity_search_with_relevance_scores(query_text, k=3)

    # If there are no results, return a message
    if len(results) == 0:
        print("Unable to find matching results.")
        return

    # Apply score normalization for cosine similarity
    normalized_results = [
        (doc, (score + 1) / 2)  # Maps score from [-1, 1] to [0, 1]
        for doc, score in results
    ]

    # Sort and get the highest score result
    best_result = max(normalized_results, key=lambda x: x[1])
    best_doc, best_score = best_result

    # Format and display the result
    prompt = PROMPT_TEMPLATE.format(label=best_doc.metadata['filename'], context=best_doc.page_content)
    print(prompt)
    print(f"Normalized similarity score: {best_score:.2f}")

if __name__ == "__main__":
    main()

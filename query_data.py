from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
import os

CHROMA_PATH = "chroma"

def query_text(query_text):
    """Queries the ChromaDB with the provided text and returns matching categories."""
    embedding_function = OpenAIEmbeddings()
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)
    
    # Perform the similarity search
    results = db.similarity_search_with_relevance_scores(query_text, k=3)
    if not results:
        print("No results found for this chunk.")
        return []
    
    # Filter results by relevance score
    filtered_results = [result for result in results if result[1] >= 0.7]
    if not filtered_results:
        print("No results with a relevance score above 0.7.")
        return []
    
    # Extract unique categories from the results
    matching_categories = [doc.metadata.get("類別", "Unknown") for doc, _ in filtered_results]
    unique_categories = list(set(matching_categories))
    
    # Debugging output
    print(f"Query Text: {query_text[:50]}...")  # Show first 50 characters for context
    print(f"Matching Categories: {unique_categories}")
    return unique_categories

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", type=str, help="The query text.")
    args = parser.parse_args()
    print(query_text(args.query_text))

import os
import pandas as pd
import numpy as np
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma  # Updated import path based on deprecation notice
from dotenv import load_dotenv
import openai 

CHROMA_PATH = "chroma"
CHUNK_CSV_PATH = "data/tcfd_report_pdf_chunks/chunk_embeddings_富邦金控_2022_300_50.csv"
OUTPUT_CSV_PATH = "data/tcfd_report_pdf_chunks_matching_result/富邦金控_2022_300_50_matched_chunks.csv"
load_dotenv()
openai.api_key = os.environ['OPENAI_API_KEY']
def load_chunks_from_csv():
    """Loads chunk embeddings and metadata from CSV."""
    return pd.read_csv(CHUNK_CSV_PATH)

def query_chroma_for_similar_chunks(embedding):
    """Queries ChromaDB and returns the top 5 results with similarity above 0.7."""
    embedding = np.array(eval(embedding)).flatten()
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=OpenAIEmbeddings(model="text-embedding-ada-002"))
    
    results = db.similarity_search_by_vector_with_relevance_scores(embedding, k=47)
    # for result in results:
    #     print(result[0])
    #     print(result[1])
    #     print(type(result))

    filtered_results = [
        {"類別": doc[0].metadata['類別'], "content": doc[0].page_content, "cosine_distance": doc[1]}
        for doc in results if doc[1] <= 0.20
    ]
    # print('filter result done')
    return filtered_results

def process_chunks_and_save():
    """Processes each chunk, queries Chroma, and saves results to a CSV."""
    df_chunks = load_chunks_from_csv()
    output_data = []

    for _, row in df_chunks.iterrows():
        file_name = row['Filename']
        embedding = row['Chunk_Embedding']
        chunk_id = row['Chunk_ID']
        chunk_text = row['Chunk_Text']
        
        # Query ChromaDB for similar chunks
        results = query_chroma_for_similar_chunks(embedding)
        # Extract unique categories from results
        matching_categories = [doc['類別'] for doc in results]
        unique_categories = list(set(matching_categories))
        cosine_distance = [doc['cosine_distance'] for doc in results]
        cosine_distance = list(set(cosine_distance))

        # Record results
        output_data.append({
            'Filename': file_name,
            'Chunk_ID': chunk_id,
            "Chunk_Text": chunk_text,
            "Embedding": embedding,
            "Matched_Categories": unique_categories,
            "Cosine_Distance": cosine_distance
        })

    # Save to CSV
    output_df = pd.DataFrame(output_data)
    output_df.to_csv(OUTPUT_CSV_PATH, index=False)
    print(f"Saved matched chunks and categories to {OUTPUT_CSV_PATH}.")

if __name__ == "__main__":
    process_chunks_and_save()

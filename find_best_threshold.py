import os
import pandas as pd
import numpy as np
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma  # Updated import path based on deprecation notice
from dotenv import load_dotenv
import openai 

CHROMA_PATH = "chroma"
CHUNK_CSV_DIRECTORY = "data/tcfd_report_pdf_chunks_第四層/"
OUTPUT_CSV_DIRECTORY = "data/tcfd_report_pdf_chunks_matching_result_第四層/"
ANSWER_PATH = "data/answer/rank.xlsx"

# Load environment variables
load_dotenv()
openai.api_key = os.environ['OPENAI_API_KEY']

# Ensure output directory exists
os.makedirs(OUTPUT_CSV_DIRECTORY, exist_ok=True)

# Initialize OpenAIEmbeddings
embedding_model = OpenAIEmbeddings(model="text-embedding-ada-002")
def load_chunks_from_csv(csv_path):
    """Loads chunk embeddings and metadata from a specific CSV file."""
    return pd.read_csv(csv_path)

# def query_chroma_for_similar_chunks(embedding):
#     """Queries ChromaDB and returns the top 5 results with similarity above 0.7."""
#     embedding = np.array(eval(embedding)).flatten()
#     db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_model)
    
#     results = db.similarity_search_by_vector_with_relevance_scores(embedding, k=47)
#     filtered_results = [
#         {"類別": doc[0].metadata['類別'], "content": doc[0].page_content, "cosine_distance": doc[1]}
#         for doc in results if doc[1] <= 0.2
#     ]
#     return filtered_results

def query_chroma_for_similar_chunks(embedding, threshold):
    """Queries ChromaDB and returns the top 5 results with similarity above 0.7."""
    # embedding = np.array(eval(embedding)).flatten()
    # db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_model)
    
    # results = db.similarity_search_by_vector_with_relevance_scores(embedding, k=47)
    # filtered_results = [
    #     {"類別": doc[0].metadata['類別'], "content": doc[0].page_content, "cosine_distance": doc[1]}
    #     for doc in results if doc[1] <= threshold
    # ]
    filter_results = [
    {
        "類別": "G-1-1",
        "content": "This is the content of the chunk.",
        "cosine_distance": 0.18
    },
    {
        "類別": "S-2-3",
        "content": "Another similar chunk's content.",
        "cosine_distance": 0.15
    }
]
    return filter_results

def process_chunks_and_save(csv_path):
    """Processes each chunk in a CSV file, queries Chroma, and saves results to a corresponding CSV."""
    df_chunks = load_chunks_from_csv(csv_path)
    output_data = []

    for _, row in df_chunks.iterrows():
        file_name = row['Filename']
        embedding = row['Chunk_Embedding']
        chunk_id = row['Chunk_ID']
        chunk_text = row['Chunk_Text']
        
        # Query ChromaDB for similar chunks
        results = query_chroma_for_similar_chunks(embedding)
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

    # Generate output file path
    base_name = os.path.basename(csv_path).replace("chunk_embeddings_", "").replace(".csv", "")
    output_file_name = f"{base_name}_matched_chunks.csv"
    output_file_path = os.path.join(OUTPUT_CSV_DIRECTORY, output_file_name)
    
    # Save to CSV
    output_df = pd.DataFrame(output_data)
    output_df.to_csv(output_file_path, index=False)
    print(f"Saved matched chunks and categories to {output_file_path}.")

def load_answer(institution, year):
    latest_answer_df = pd.read_excel(ANSWER_PATH)
    if(institution[2:4] == "金控"):
        institution = institution[0:2] + "金"
    # print(latest_answer_df)
    # print(institution)
    year = int(year)
    answer_for_institution = latest_answer_df[latest_answer_df['Financial_Institutions'] == institution]
    # print(f"Unique values in latest_answer_df['Year']: {latest_answer_df['Year'].unique()}")
    # print(f"Type of 'Year' column: {latest_answer_df['Year'].dtype}")
    # print(f"Value of year: {year}, type: {type(year)}")
    answer_for_institution = answer_for_institution[latest_answer_df['Year'] == year]
    # print(answer_for_institution)
    columns_to_print = answer_for_institution.loc[:, "Q1":"Q82"].iloc[0].to_dict()
    # print(answer_for_institution.loc[:, "Q1":"Q82"])
    return columns_to_print

def read_report_pdf(csv_path):
    df_chunks = load_chunks_from_csv(csv_path)
    output_data = []

    for _, row in df_chunks.iterrows():
        file_name = row['Filename']
        embedding = row['Chunk_Embedding']
        chunk_id = row['Chunk_ID']
        chunk_text = row['Chunk_Text']
        output_data.append({
                'Filename': file_name,
                'Chunk_ID': chunk_id,
                "Chunk_Text": chunk_text,
                "Embedding": embedding,
        })
    return output_data

def calculate_accuracy():
    return

def optimize_threshold(answer, report_dict):
    chunk_csv_files = [
        os.path.join(CHUNK_CSV_DIRECTORY, f) 
        for f in os.listdir(CHUNK_CSV_DIRECTORY) if f.endswith('.csv')
    ]
    print(chunk_csv_files)
    thresholds = np.arange(0.0, 2.1, 0.1)
    accuracy_in_threshold = []
    for threshold in thresholds:
        for csv_path in chunk_csv_files:
            print(f"\nProcessing {csv_path}...")
            institution = csv_path.split('/')[2].split('_')[2]
            year = csv_path.split('/')[2].split('_')[3]
            print(query_chroma_for_similar_chunks(read_report_pdf(csv_path)[0].get('Embedding'), 1.0))

            # print(institution[2:4])
            # print(load_answer(institution, year))
            # print(read_report_pdf(csv_path)[0].get('Embedding'))
            # process_chunks_and_save(csv_path)
    return accuracy_in_threshold

def main():
    optimize_threshold()
    

if __name__ == "__main__":
    main()

from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from transformers import AutoTokenizer, AutoModel
import os
import shutil
import pandas as pd

# Constants
EXCEL_FILE_PATH = 'data/tcfd/銀行業名稱_年份_TCFD判讀(報告書_網頁).xlsx'
CHROMA_PATH = "chroma"
DATA_PATH = "data/tcfd"  # Directory to store extracted text files

# Hugging Face Model for Embeddings
tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")
model = AutoModel.from_pretrained("bert-base-chinese")

def generate_embeddings(text: str):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).detach().numpy()

def create_huggingface_embeddings():
    return HuggingFaceEmbeddings(model_name="bert-base-chinese")

def main():
    extract_and_save_guidelines()
    generate_data_store()

def extract_and_save_guidelines():
    # Load the Excel file and process the '判讀結果' sheet
    xls = pd.ExcelFile(EXCEL_FILE_PATH)
    df_judgment = pd.read_excel(xls, sheet_name='判讀結果')
    
    # Extract rows that contain third-level guidelines from 'Unnamed: 2'
    third_layer_df = df_judgment[df_judgment['Unnamed: 2'].notna() & df_judgment['Unnamed: 2'].str.startswith(('G-', 'S-', 'R-', 'MT-', '#'))]
    
    # Save each guideline into a text file in the output directory
    if not os.path.exists(DATA_PATH):
        os.makedirs(DATA_PATH)

    for index, row in third_layer_df.iterrows():
        guideline_code = row['Unnamed: 2'].split('\n')[0]  # Extract the guideline code (e.g., G-1-1, S-1-1, etc.)
        content = row['Unnamed: 2']  # Full content for the guideline
        file_path = os.path.join(DATA_PATH, f"{guideline_code}.txt")
        
        # Save the content into a text file
        with open(file_path, 'w', encoding='utf-8') as file:
            file.write(content)

    print(f"Extracted and saved {len(third_layer_df)} guidelines to {DATA_PATH}.")

def generate_data_store():
    documents = load_documents()
    chunks = split_text(documents)
    save_to_chroma(chunks)

def load_documents():
    loader = DirectoryLoader(DATA_PATH, glob="*.txt")
    documents = loader.load()
    return documents

def split_text(documents: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=100,
        length_function=len,
        add_start_index=True,
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Split {len(documents)} documents into {len(chunks)} chunks.")
    return chunks

def save_to_chroma(chunks: list[Document]):
    # Clear out the database first.
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)

    # Create a new DB from the documents using Hugging Face embeddings
    hf_embeddings = create_huggingface_embeddings()
    
    db = Chroma.from_documents(
        chunks, hf_embeddings, persist_directory=CHROMA_PATH
    )
    db.persist()
    print(f"Saved {len(chunks)} chunks to {CHROMA_PATH}.")

if __name__ == "__main__":
    main()
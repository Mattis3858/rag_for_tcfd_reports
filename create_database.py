from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
# from langchain.embeddings import OpenAIEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
import openai 
from dotenv import load_dotenv
import os
import shutil
import pandas as pd

# Load environment variables. Assumes that project contains .env file with API keys
load_dotenv()
#---- Set OpenAI API key 
# Change environment variable name from "OPENAI_API_KEY" to the name given in 
# your .env file.
openai.api_key = os.environ['OPENAI_API_KEY']

CHROMA_PATH = "chroma"
EXCEL_PATH = "data/tcfd接露指引 - 加上第一層指標及關鍵字.xlsx"


def main():
    generate_data_store()


def generate_data_store():
    documents = load_documents_from_excel()
    if documents:  # Only proceed if we have valid documents
        chunks = split_text(documents)
        save_to_chroma(chunks)
    else:
        print("No valid documents were loaded. Please check your Excel file.")


def load_documents_from_excel():
    try:
        # Load the Excel file
        df = pd.read_excel(EXCEL_PATH)
        
        # Print column names to help with debugging
        print("Available columns:", df.columns.tolist())
        
        documents = []
        for idx, row in df.iterrows():
            # Get the text content and handle NaN values
            text = row['第三層(TCFD) 揭露指引']
            category = row['類別']
            
            # Skip rows where either text or category is NaN/empty
            if pd.isna(text) or pd.isna(category):
                print(f"Skipping row {idx + 2} due to missing data")
                continue
            
            # Convert to string and strip whitespace
            text = str(text).strip()
            category = str(category).strip()
            
            # Only add non-empty documents
            if text and category:
                metadata = {'類別': category}
                try:
                    documents.append(Document(page_content=text, metadata=metadata))
                except Exception as e:
                    print(f"Error creating document for row {idx + 2}: {e}")
            else:
                print(f"Skipping row {idx + 2} due to empty content after cleaning")
        
        print(f"Successfully loaded {len(documents)} valid documents")
        return documents
    
    except FileNotFoundError:
        print(f"Excel file not found at path: {EXCEL_PATH}")
        return []
    except Exception as e:
        print(f"Error loading Excel file: {e}")
        return []

def split_text(documents: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=8000,
        chunk_overlap=4000,
        length_function=len,
        add_start_index=True,
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Split {len(documents)} documents into {len(chunks)} chunks.")

    document = chunks[10] if len(chunks) > 10 else chunks[0]
    print(document.page_content)
    print(document.metadata)

    return chunks


def save_to_chroma(chunks: list[Document]):
    # Clear out the database first.
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)

    # Create a new DB from the documents.
    db = Chroma.from_documents(
        chunks, OpenAIEmbeddings(model="text-embedding-ada-002"), persist_directory=CHROMA_PATH
    )
    db.persist()
    print(f"Saved {len(chunks)} chunks to {CHROMA_PATH}.")


if __name__ == "__main__":
    main()

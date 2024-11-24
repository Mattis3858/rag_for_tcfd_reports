import os
import pandas as pd
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
import openai
from dotenv import load_dotenv
import pymupdf

# 定義 PDF 檔案所在的資料夾
PDF_DIRECTORY = "data/tcfd_report_pdf_for_testing/"
OUTPUT_CSV = "data/tcfd_report_pdf_chunks/chunk_embeddings_富邦金控_2022_300_50_proprocessed.csv"
load_dotenv()
openai.api_key = os.environ['OPENAI_API_KEY']
# 初始化 OpenAIEmbeddings
embedding_model = OpenAIEmbeddings(model="text-embedding-ada-002")

def load_pdf(file_path):
    """Reads PDF and extracts text."""
    doc = pymupdf.open(file_path)
    text = ""
    for page in doc:
        text += page.get_text() or ""
        print(text)
    return text

def split_pdf_text(text):
    """Splits text from PDF into smaller chunks."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=50,
        length_function=len,
        add_start_index=True,
    )
    return text_splitter.split_text(text)

def get_chunk_embedding(chunk_text):
    """Gets embedding vector for a text chunk using OpenAIEmbeddings."""
    return embedding_model.embed_query(chunk_text)  # 使用 langchain_openai 的嵌入模型

def process_pdf_files(file_paths):
    """Processes each PDF file and saves chunk embeddings to CSV."""
    data = []
    for file_path in file_paths:
        print(f"\nProcessing {file_path}...")
        text = load_pdf(file_path)
        chunks = split_pdf_text(text)
        
        for idx, chunk_text in enumerate(chunks):
            print(f"Processing chunk {idx + 1}/{len(chunks)}")
            chunk_embedding = get_chunk_embedding(chunk_text)
            
            # 將資料儲存到字典中
            data.append({
                "Filename": os.path.basename(file_path),
                "Chunk_ID": idx,
                "Chunk_Text": chunk_text,
                "Chunk_Embedding": chunk_embedding
            })

    # 將結果儲存到 DataFrame，並輸出到 CSV
    result_df = pd.DataFrame(data)
    result_df.to_csv(OUTPUT_CSV, index=False)
    print(f"Results saved to {OUTPUT_CSV}.")

def main():
    # 自動偵測資料夾中的所有 PDF 檔案
    pdf_files = [os.path.join(PDF_DIRECTORY, f) for f in os.listdir(PDF_DIRECTORY) if f.endswith('.pdf')]
    
    # 開始處理這些 PDF 檔案
    process_pdf_files(pdf_files)

if __name__ == "__main__":
    main()

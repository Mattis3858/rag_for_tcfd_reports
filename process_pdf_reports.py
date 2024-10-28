import os
import argparse
import pandas as pd
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import query_data  # Import the modified query_data module


# 定義 PDF 檔案所在的資料夾
PDF_DIRECTORY = "data/tcfd_report_pdf/"
CHROMA_PATH = "chroma"
file_path = 'data/tcfd接露指引.xlsx'  # Update path as needed
tcfd_data = pd.read_excel(file_path)
CATEGORY_COLUMNS = tcfd_data['類別'].tolist()

def load_pdf(file_path):
    """Reads PDF and extracts text."""
    reader = PdfReader(file_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    return text

def split_pdf_text(text):
    """Splits text from PDF into smaller chunks."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=100,
        length_function=len,
        add_start_index=True,
    )
    return text_splitter.split_text(text)

def query_chunks(chunks):
    """Queries each chunk and aggregates unique categories."""
    unique_categories = set()
    for idx, chunk in enumerate(chunks):
        # Use the query_text function to retrieve categories
        print(f"Querying chunk {idx + 1}/{len(chunks)}")
        chunk_categories = query_data.query_text(chunk)
        print(f"Categories found for chunk {idx + 1}: {chunk_categories}")
        unique_categories.update(chunk_categories)
    print(f"Unique categories for document: {unique_categories}")
    return unique_categories

def process_pdf_files(file_paths):
    """Processes each PDF file and aggregates the categories."""
    data = []
    for file_path in file_paths:
        print(f"\nProcessing {file_path}...")
        # Load PDF text and split into chunks
        text = load_pdf(file_path)
        chunks = split_pdf_text(text)
        
        # Query each chunk and aggregate unique categories
        unique_categories = query_chunks(chunks)
        
        # Initialize category row
        row = {category: 0 for category in CATEGORY_COLUMNS}
        row['Filename'] = os.path.basename(file_path)
        
        # Mark identified categories as 1
        for category in unique_categories:
            if category in row:
                row[category] = 1  # Use the exact category name from unique_categories
        
        # Debugging output for the row
        print(f"Data row for {file_path}: {row}")
        
        data.append(row)
    result_df = pd.DataFrame(data)
    
    # Reorder columns to have 'Filename' as the first column
    columns_order = ['Filename'] + CATEGORY_COLUMNS
    result_df = result_df[columns_order]  # Reorder DataFrame columns
    
    # Save to CSV
    result_df.to_csv("aggregated_results__03_03_300.csv", index=False)
    print("Results saved to aggregated_results.csv.")
    
    return result_df

def main():
    # 自動偵測資料夾中的所有 PDF 檔案
    pdf_files = [os.path.join(PDF_DIRECTORY, f) for f in os.listdir(PDF_DIRECTORY) if f.endswith('.pdf')]
    
    # 開始處理這些 PDF 檔案
    process_pdf_files(pdf_files)

if __name__ == "__main__":
    main()

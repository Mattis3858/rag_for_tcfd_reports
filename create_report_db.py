import os
import shutil
from dotenv import load_dotenv
import openai
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma

# 載入環境變數以取得 OpenAI API 金鑰
load_dotenv()
openai.api_key = os.getenv('OPENAI_API_KEY')

# 定義設定
BASE_CHROMA_PATH = "chroma_tcfd"
CHUNK_SIZE = 300
CHUNK_OVERLAP = 100

def process_pdf(pdf_path):
    # 使用報告書檔案名稱作為子資料夾名稱
    pdf_name = os.path.splitext(os.path.basename(pdf_path))[0]
    chroma_path = os.path.join(BASE_CHROMA_PATH, pdf_name)

    # 載入並將 PDF 分割成塊狀
    loader = PyPDFLoader(pdf_path)
    text_splitter = RecursiveCharacterTextSplitter(
        separators="\n",
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )
    documents = loader.load_and_split(text_splitter)
    print(f"{pdf_name} 分割後的塊數量：{len(documents)}")

    # 如果已存在該報告書的 ChromaDB 資料夾，則清除並重新建立
    if os.path.exists(chroma_path):
        shutil.rmtree(chroma_path)
    
    # 建立嵌入向量並儲存在 ChromaDB 中
    db = Chroma.from_documents(
        documents=documents,
        embedding=OpenAIEmbeddings(),
        persist_directory=chroma_path
    )
    db.persist()
    print(f"{pdf_name} 的 ChromaDB 已建立並儲存至 '{chroma_path}'")

def main():
    pdf_paths = [
        
        "暫存\\2893_新光銀行_2021_TCFD_專章完整永續_preprocessed.pdf",
        "暫存\\6039_將來銀行_2022_TCFD_報告書_preprocessed.pdf"
        
        
    ]
    
    for pdf_path in pdf_paths:
        process_pdf(pdf_path)

if __name__ == "__main__":
    main()

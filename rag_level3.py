import os
import pandas as pd
from dotenv import load_dotenv
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma

# 加載環境變數以取得 OpenAI API 金鑰
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

# 設定文件路徑與 ChromaDB 的基礎資料夾路徑
EXCEL_PATH = "C:\\Users\\bugee\\OneDrive\\桌面\\RAG\\rag_for_tcfd_reports\\data\\tcfd接露指引 - 加上第一層指標及關鍵字.xlsx"
BASE_CHROMA_PATH = "chroma_tcfd"
OUTPUT_PATH = "C:\\Users\\bugee\\OneDrive\\桌面\\RAG\\rag_for_tcfd_reports\\data\\label_result"

# 確保輸出目錄存在
os.makedirs(OUTPUT_PATH, exist_ok=True)

def load_documents_from_excel():
    # 從 Excel 文件載入指標名稱和定義
    df = pd.read_excel(EXCEL_PATH)
    documents = []
    for _, row in df.iterrows():
        text = row['第三層(TCFD) 揭露指引']
        metadata = {'類別': row['類別']}
        documents.append((text, metadata))
    return documents

def query_text(query_text, pointer_name, chroma_path):
    # 查詢指定 ChromaDB 並返回匹配的類別
    embedding_function = OpenAIEmbeddings()
    db = Chroma(persist_directory=chroma_path, embedding_function=embedding_function)
    
    results = db.similarity_search_with_relevance_scores(query_text, k=100)
    filtered_results = [result for result in results if result[1] >= 0.8]
    
    # 建立儲存結果的 DataFrame
    result_data = []
    for doc, score in filtered_results:
        result_data.append({
            "指標名稱": pointer_name,
            "相似度": score,
            "片段內容": doc.page_content
        })
    
    return result_data

def process_report(report_name):
    chroma_path = os.path.join(BASE_CHROMA_PATH, report_name)
    documents = load_documents_from_excel()
    all_results = []

    for text, metadata in documents:
        pointer_name = metadata['類別']
        definition = text
        query = f'此報告書是否揭露 "{pointer_name}"? "{pointer_name}" 定義如下：{definition}'

        # 執行相似度查詢並收集結果
        result_data = query_text(query, pointer_name, chroma_path)
        all_results.extend(result_data)

    # 將每份報告書的結果儲存到單獨的 CSV 檔案
    output_csv_path = os.path.join(OUTPUT_PATH, f"label_result_{report_name}.csv")
    output_df = pd.DataFrame(all_results)
    output_df.to_csv(output_csv_path, index=False, encoding="utf-8-sig")
    print(f"結果已儲存到：{output_csv_path}")

def main():
    report_names = [
       "富邦金控_2022_TCFD_報告書_preprocessed"
    ]
    
    for report_name in report_names:
        process_report(report_name)

if __name__ == "__main__":
    main()

import os
import pandas as pd
from dotenv import load_dotenv
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma

# 加載環境變數以取得 OpenAI API 金鑰
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
print(f"[DEBUG] Loaded API Key: {api_key}")

# 設定文件路徑與 ChromaDB 的基礎資料夾路徑
EXCEL_PATH = "C:\\Users\\bugee\\OneDrive\\桌面\\RAG\\rag_for_tcfd_reports\\data\\tcfd接露指引 - 加上第一層指標.xlsx"
BASE_CHROMA_PATH = "chroma_tcfd"
OUTPUT_PATH = "C:\\Users\\bugee\\OneDrive\\桌面\\RAG\\rag_for_tcfd_reports\\data\\label_result"

# 確保輸出目錄存在
os.makedirs(OUTPUT_PATH, exist_ok=True)
print(f"[DEBUG] OUTPUT_PATH exists or created: {OUTPUT_PATH}")

def load_documents_from_excel():
    print("[DEBUG] Loading documents from Excel...")
    try:
        # 從 Excel 文件載入指標名稱和定義
        df = pd.read_excel(EXCEL_PATH)
        print(f"[DEBUG] Excel loaded successfully with {len(df)} rows.")
        documents = []
        for _, row in df.iterrows():
            text = row['第三層(TCFD) 揭露指引']
            metadata = {'類別': row['類別']}
            documents.append((text, metadata))
        print(f"[DEBUG] Loaded {len(documents)} documents.")
        return documents
    except Exception as e:
        print(f"[ERROR] Failed to load documents from Excel: {e}")
        return []

def query_text(query_text, pointer_name, chroma_path):
    print(f"[DEBUG] Querying text for pointer_name: {pointer_name}")
    try:
        embedding_function = OpenAIEmbeddings()
        db = Chroma(persist_directory=chroma_path, embedding_function=embedding_function)
        print(f"[DEBUG] Connected to ChromaDB at {chroma_path}")

        results = db.similarity_search_with_relevance_scores(query_text, k=100)
        print(f"[DEBUG] Retrieved {len(results)} results.")
        
        # 篩選出相關度 >= 0.82 的結果並取前 10 個
        filtered_results = [result for result in results if result[1] >= 0.835]
        filtered_results = sorted(filtered_results, key=lambda x: x[1], reverse=True)[:5]
        print(f"[DEBUG] Filtered and sorted top {len(filtered_results)} results with relevance >= 0.8.")
        
        # 建立儲存結果的 DataFrame
        result_data = []
        for doc, score in filtered_results:
            result_data.append({
                "指標名稱": pointer_name,
                "相似度": score,
                "片段內容": doc.page_content
            })
        
        return result_data
    except Exception as e:
        print(f"[ERROR] Failed to query text: {e}")
        return []

def process_report(report_name):
    print(f"[DEBUG] Processing report: {report_name}")
    chroma_path = os.path.join(BASE_CHROMA_PATH, report_name)
    print(f"[DEBUG] Chroma path set to: {chroma_path}")

    documents = load_documents_from_excel()
    if not documents:
        print("[ERROR] No documents loaded. Skipping report.")
        return

    all_results = []
    for text, metadata in documents:
        pointer_name = metadata['類別']
        definition = text
        query = f'此報告書是否揭露 "{pointer_name}"? "{pointer_name}" 定義如下：{definition}'

        # 執行相似度查詢並收集結果
        print(f"[DEBUG] Querying for pointer_name: {pointer_name}")
        result_data = query_text(query, pointer_name, chroma_path)
        all_results.extend(result_data)

    # 將每份報告書的結果儲存到單獨的 CSV 檔案
    output_csv_path = os.path.join(OUTPUT_PATH, f"label_result_{report_name}.csv")
    print(f"[DEBUG] Saving results to {output_csv_path}")
    try:
        output_df = pd.DataFrame(all_results)
        output_df.to_csv(output_csv_path, index=False, encoding="utf-8-sig")
        print(f"[DEBUG] Results saved to: {output_csv_path}")
    except Exception as e:
        print(f"[ERROR] Failed to save results: {e}")

def main():
    print("[DEBUG] Starting main process...")
    report_names = [
        # # "6031_連線銀行_2022_TCFD_報告書",
        "2893_新光銀行_2021_TCFD_專章完整永續_preprocessed"
        # # "6039_將來銀行_2022_TCFD_報告書_preprocessed",
        # "富邦金控_2022_TCFD_報告書_preprocessed"
    ]
    
    for report_name in report_names:
        process_report(report_name)
    print("[DEBUG] All reports processed.")

if __name__ == "__main__":
    main()

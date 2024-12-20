import os
import time
import psutil
import pandas as pd
from dotenv import load_dotenv
import os
import pandas as pd
from dotenv import load_dotenv
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

# 載入環境變數
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

# 檔案路徑
EXCEL_PATH = r"C:\Users\bugee\OneDrive\桌面\RAG\rag_for_tcfd_reports\data\tcfd第四層接露指引 - 加上第一層指標及關鍵字.xlsx"
RANK_PATH = r"C:\Users\bugee\OneDrive\桌面\RAG\rag_for_tcfd_reports\data\answer\rank.xlsx"

BASE_CHROMA_PATH = "chroma_tcfd"
OUTPUT_PATH = "rag_for_tcfd_reports/data/label_result"
ACCURACY_RESULT_BASE = "rag_for_tcfd_reports/data/accuracy_result"
ALL_RESULTS_PATH_TEMPLATE = r"C:\Users\bugee\OneDrive\桌面\query_result\all_query_results_{}.csv"

# 確保基礎目錄存在
os.makedirs(OUTPUT_PATH, exist_ok=True)
os.makedirs(ACCURACY_RESULT_BASE, exist_ok=True)

def map_label_to_q(label: str) -> str:
    last_part = label.split('_')[-1]
    if last_part.startswith("#MT"):
        replaced = last_part.replace("#MT", "MT", 1)
        return f"Q{replaced}"
    elif last_part.startswith("#S"):
        replaced = last_part.replace("#S", "S", 1)
        return f"Q{replaced}"
    elif last_part.isdigit():
        return f"Q{last_part}"
    else:
        return label

def load_documents():
    df = pd.read_excel(EXCEL_PATH)
    documents = []
    for idx, row in df.iterrows():
        text = row.get('第四層(TCFD) 揭露指引')
        label = row.get('類別')
        if pd.isna(text) or pd.isna(label):
            print(f"[WARN] 檔案中存在缺失的文本或標籤。行號: {idx}")
            continue
        documents.append((text, label))
    print(f"[INFO] 共載入 {len(documents)} 個文件。")
    return documents

def get_all_results(report_name, k=50):
    chroma_path = os.path.join(BASE_CHROMA_PATH, report_name)
    documents = load_documents()
    if not documents:
        print(f"[ERROR] {report_name} 的文件列表為空。")
        return pd.DataFrame()
    embedding_function = OpenAIEmbeddings()
    db = Chroma(persist_directory=chroma_path, embedding_function=embedding_function)
    ALL_RESULTS_PATH = ALL_RESULTS_PATH_TEMPLATE.format(report_name)

    if os.path.exists(ALL_RESULTS_PATH):
        try:
            df_all = pd.read_csv(ALL_RESULTS_PATH)
            if df_all.empty:
                print(f"[WARN] {ALL_RESULTS_PATH} 是空的，將重新生成結果。")
            else:
                print(f"[INFO] 從 {ALL_RESULTS_PATH} 載入預計算結果")
                return df_all
        except pd.errors.EmptyDataError:
            print(f"[WARN] {ALL_RESULTS_PATH} 是空的或格式錯誤，將重新生成結果。")

    # 如果文件不存在或是空的，則重新生成
    all_results = []
    total_docs = len(documents)
    print(f"[INFO] 開始檢索 {report_name} 的所有結果：{total_docs} 個文件 (k={k})")

    start_time = time.time()
    log_interval = 10  # 每處理 10 個文件打印一次進度

    for i, (text, original_label) in enumerate(documents, start=1):
        mapped_label = map_label_to_q(original_label)
        query = f'揭露指標：{original_label}, 定義如下：{text}'
        try:
            results = db.similarity_search_with_relevance_scores(query, k=k)
        except Exception as e:
            print(f"[ERROR] {report_name}: 查詢過程中發生錯誤：{e}")
            continue

        # 每 log_interval 個文件打印一次進度
        if i % log_interval == 0:
            elapsed_time = time.time() - start_time
            cpu_usage = psutil.cpu_percent()
            mem_info = psutil.virtual_memory()
            memory_usage = mem_info.used / (1024**3)  # 轉換為 GB
            print(f"[INFO] {report_name}: 已處理 {i}/{total_docs} 個文件...")
            print(f"[INFO] 已用時間: {elapsed_time:.2f}s, CPU 使用率: {cpu_usage}%, 記憶體使用量: {memory_usage:.2f}GB")

        if not results:
            print(f"[WARN] {report_name}: 文件 {i} 沒有找到相關結果。")
            continue
        else:
            print(f"[INFO] {report_name}: 文件 {i} 返回 {len(results)} 個相關結果。")

        for doc, score in results:
            all_results.append({
                "original_label": original_label,
                "mapped_label": mapped_label,
                "score": score,
                "content": doc.page_content
            })

    if not all_results:
        print(f"[WARN] {report_name} 沒有生成任何結果。")
        return pd.DataFrame()  # 返回空的 DataFrame

    df_all = pd.DataFrame(all_results)
    df_all.to_csv(ALL_RESULTS_PATH, index=False, encoding="utf-8-sig")
    print(f"[INFO] {report_name}: 所有查詢結果已保存至: {ALL_RESULTS_PATH}")
    return df_all

def calculate_accuracy_for_report(df_all, threshold, rank_file, institution, year, report_name):
    # 根據指定的閾值篩選文件並計算準確率
    filtered = df_all[df_all["score"] >= threshold]
    predicted_labels = set(filtered["mapped_label"].unique())

    # 載入真實標籤
    rank_df = pd.read_excel(rank_file)
    ground_truth_df = rank_df[(rank_df['Financial_Institutions'] == institution) &
                              (rank_df['Year'] == year)]
    
    if ground_truth_df.empty:
        print(f"[WARN] 找不到 {institution} 在 {year} 年的真實標籤。跳過。")
        return None

    ground_truth = ground_truth_df.loc[:, "Q1":"Q82"].iloc[0].to_dict()

    ground_truth_list = []
    predicted_list = []

    for key, value in ground_truth.items():
        ground_truth_value = int(value) if not pd.isna(value) else 0
        predicted_value = 1 if key in predicted_labels else 0
        ground_truth_list.append(ground_truth_value)
        predicted_list.append(predicted_value)

    accuracy = sum(gt == pred for gt, pred in zip(ground_truth_list, predicted_list)) / len(ground_truth_list)
    return accuracy

def main():
    # 定義六份報告的相關資訊 (報告名稱, 機構, 年份)
    report_info_list = [
        ("富邦金控_2022_TCFD_報告書_preprocessed", "富邦金", 2022),
        ("新光金控_2022_TCFD_報告書_preprocessed", "新光金", 2022),
        ("永豐銀行_2022_TCFD_報告書_preprocessed", "永豐銀行", 2022),
        ("開發金控_2021_TCFD_報告書_preprocesed", "開發金", 2021),  # 保持不變
        ("瑞興銀行_2022_TCFD_報告書", "瑞興銀行", 2022),
        ("華泰銀行_2022_TCFD_報告書_preprocessed", "華泰銀行", 2022)
    ]

    # 載入所有報告的資料
    all_reports_data = []
    for report_name, institution, year in report_info_list:
        print(f"\n[INFO] 處理報告: {report_name}")
        df_all = get_all_results(report_name, k=50)
        if df_all.empty:
            print(f"[ERROR] {report_name} 沒有生成任何結果，請檢查數據或查詢邏輯。")
        all_reports_data.append({
            "report_name": report_name,
            "institution": institution,
            "year": year,
            "df_all": df_all
        })

    # 定義以 0.01 為增量的閾值
    thresholds = [x / 100 for x in range(75, 96, 1)]  # 0.75, 0.76, ... 0.95
    best_threshold = None
    best_average_accuracy = -1.0

    print("\n[INFO] 開始閾值優化...")

    for threshold in thresholds:
        accuracies = []
        print(f"\n[INFO] 測試閾值: {threshold}")
        for report_data in all_reports_data:
            report_name = report_data["report_name"]
            institution = report_data["institution"]
            year = report_data["year"]
            df_all = report_data["df_all"]

            if df_all.empty:
                print(f"[WARN] {report_name} 的數據為空，跳過此報告的準確率計算。")
                continue

            acc = calculate_accuracy_for_report(df_all, threshold, RANK_PATH, institution, year, report_name)
            if acc is not None:
                accuracies.append(acc)
                print(f"[INFO] {report_name}: 閾值 {threshold} 下的準確率 = {acc:.2%}")
            else:
                print(f"[WARN] {report_name}: 無法計算準確率，跳過。")

        if not accuracies:
            print("[ERROR] 此閾值下沒有計算出任何準確率。跳過此閾值。")
            continue

        avg_acc = sum(accuracies) / len(accuracies)
        print(f"[INFO] 閾值: {threshold}, 平均準確率: {avg_acc:.2%}")

        if avg_acc > best_average_accuracy:
            best_average_accuracy = avg_acc
            best_threshold = threshold

    if best_threshold is None:
        print("[ERROR] 沒有找到有效的閾值。結束程式。")
        return

    print(f"\n[INFO] 所有六份報告的最佳閾值: {best_threshold}, 平均準確率: {best_average_accuracy:.2%}")

    # 使用最佳閾值計算並保存每份報告的準確率
    summary_results = []
    for report_data in all_reports_data:
        report_name = report_data["report_name"]
        institution = report_data["institution"]
        year = report_data["year"]
        df_all = report_data["df_all"]

        if df_all.empty:
            print(f"[WARN] {report_name} 的數據為空，跳過此報告的準確率計算。")
            continue

        acc = calculate_accuracy_for_report(df_all, best_threshold, RANK_PATH, institution, year, report_name)
        if acc is not None:
            summary_results.append({
                "report_name": report_name,
                "institution": institution,
                "year": year,
                "best_threshold": best_threshold,
                "accuracy_at_best_threshold": acc
            })
            print(f"[INFO] {report_name}: 在最佳閾值 {best_threshold} 下的準確率 = {acc:.2%}")
        else:
            print(f"[WARN] {report_name}: 無法計算準確率，跳過總結。")

    if summary_results:
        summary_df = pd.DataFrame(summary_results)
        summary_path = os.path.join(ACCURACY_RESULT_BASE, "summary_of_all_reports_best_threshold.csv")
        summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")
        print(f"[INFO] 含最佳閾值的總結已保存至: {summary_path}")
    else:
        print("[WARN] 沒有生成任何總結結果。")

if __name__ == "__main__":
    main()

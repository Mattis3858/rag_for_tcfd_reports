import os
import time
import psutil
import pandas as pd
from dotenv import load_dotenv
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

# Load environment variables
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

# Paths to files
EXCEL_PATH = r"C:\Users\bugee\OneDrive\桌面\RAG\rag_for_tcfd_reports\data\tcfd第四層接露指引 - 加上第一層指標及關鍵字.xlsx"
RANK_PATH = r"C:\Users\bugee\OneDrive\桌面\RAG\rag_for_tcfd_reports\data\answer\rank.xlsx"

BASE_CHROMA_PATH = "chroma_tcfd"
OUTPUT_PATH = "rag_for_tcfd_reports/data/label_result"
ACCURACY_RESULT_BASE = "rag_for_tcfd_reports/data/accuracy_result"
ALL_RESULTS_PATH_TEMPLATE = r"C:\Users\bugee\OneDrive\桌面\query_result\all_query_results_{}.csv"

# Ensure base directories exist
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
    for _, row in df.iterrows():
        text = row['第四層(TCFD) 揭露指引']
        label = row['類別']
        documents.append((text, label))
    return documents

def get_all_results(report_name, k=50):
    chroma_path = os.path.join(BASE_CHROMA_PATH, report_name)
    documents = load_documents()
    embedding_function = OpenAIEmbeddings()
    db = Chroma(persist_directory=chroma_path, embedding_function=embedding_function)
    ALL_RESULTS_PATH = ALL_RESULTS_PATH_TEMPLATE.format(report_name)

    if os.path.exists(ALL_RESULTS_PATH):
        print(f"[INFO] Loading precomputed results from {ALL_RESULTS_PATH}")
        return pd.read_csv(ALL_RESULTS_PATH)

    all_results = []
    total_docs = len(documents)
    print(f"[INFO] Start retrieving all results for {report_name}: {total_docs} documents (k={k})")

    start_time = time.time()
    log_interval = 50  # 每50個文件印出一次狀態

    for i, (text, original_label) in enumerate(documents, start=1):
        mapped_label = map_label_to_q(original_label)
        query = f'揭露指標：{original_label}, 定義如下：{text}'
        results = db.similarity_search_with_relevance_scores(query, k=k)

        # 每50筆文件印出中間狀態
        if i % log_interval == 0:
            elapsed_time = time.time() - start_time
            cpu_usage = psutil.cpu_percent()
            mem_info = psutil.virtual_memory()
            memory_usage = mem_info.used / (1024**3)  # 轉換成GB
            print(f"[INFO] {report_name}: Processed {i}/{total_docs} documents...")
            print(f"[INFO] Elapsed Time: {elapsed_time:.2f}s, CPU Usage: {cpu_usage}%, Memory Usage: {memory_usage:.2f}GB")

        for doc, score in results:
            all_results.append({
                "original_label": original_label,
                "mapped_label": mapped_label,
                "score": score,
                "content": doc.page_content
            })

    df_all = pd.DataFrame(all_results)
    df_all.to_csv(ALL_RESULTS_PATH, index=False, encoding="utf-8-sig")
    print(f"[INFO] {report_name}: All query results saved to: {ALL_RESULTS_PATH}")
    return df_all

def calculate_accuracy_for_report(df_all, threshold, rank_file, institution, year, report_name):
    # 基於指定 threshold 從 df_all 篩選文件並計算該報告書 accuracy
    filtered = df_all[df_all["score"] >= threshold]
    predicted_labels = set(filtered["mapped_label"].unique())

    rank_df = pd.read_excel(rank_file)
    ground_truth_df = rank_df[(rank_df['Financial_Institutions'] == institution) &
                              (rank_df['Year'] == year)]
    
    if ground_truth_df.empty:
        print(f"[WARN] No ground truth found for {institution} in {year}. Skipping.")
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
    # 定義六份報告書的相關資訊 (report_name, institution, year)
    report_info_list = [
        ("富邦金控_2022_TCFD_報告書_preprocessed", "富邦金", 2022),
        ("XXX金控_2022_TCFD_報告書_preprocessed", "XXX金", 2022),
        ("YYY金控_2021_TCFD_報告書_preprocessed", "YYY金", 2021),
        ("ZZZ金控_2020_TCFD_報告書_preprocessed", "ZZZ金", 2020),
        ("AAA金控_2022_TCFD_報告書_preprocessed", "AAA金", 2022),
        ("BBB金控_2022_TCFD_報告書_preprocessed", "BBB金", 2022)
    ]

    # 先載入全部報告書的結果
    all_reports_data = []
    for report_name, institution, year in report_info_list:
        print(f"\n[INFO] Processing report: {report_name}")
        df_all = get_all_results(report_name, k=50)
        all_reports_data.append({
            "report_name": report_name,
            "institution": institution,
            "year": year,
            "df_all": df_all
        })

    # 我們要在這六份報告書上找出一個單一 threshold，使得所有報告書的平均 accuracy 最佳
    thresholds = [x / 100 for x in range(75, 96, 2)]  # 0.75, 0.77, ... 0.95
    best_threshold = None
    best_average_accuracy = -1.0

    print("\n[INFO] Starting threshold optimization...")

    for threshold in thresholds:
        accuracies = []
        print(f"\n[INFO] Testing threshold: {threshold}")
        for report_data in all_reports_data:
            report_name = report_data["report_name"]
            institution = report_data["institution"]
            year = report_data["year"]
            df_all = report_data["df_all"]

            acc = calculate_accuracy_for_report(df_all, threshold, RANK_PATH, institution, year, report_name)
            if acc is not None:
                accuracies.append(acc)
                print(f"[INFO] {report_name}: Accuracy at threshold {threshold} = {acc:.2%}")
            else:
                print(f"[WARN] {report_name}: No ground truth available. Skipping accuracy calculation.")

        if not accuracies:
            print("[ERROR] No accuracies calculated for this threshold. Skipping to next threshold.")
            continue

        avg_acc = sum(accuracies) / len(accuracies)
        print(f"[INFO] Threshold: {threshold}, Average Accuracy: {avg_acc:.2%}")

        if avg_acc > best_average_accuracy:
            best_average_accuracy = avg_acc
            best_threshold = threshold

    if best_threshold is None:
        print("[ERROR] No valid thresholds found. Exiting.")
        return

    print(f"\n[INFO] Best threshold across all 6 reports: {best_threshold}, Average Accuracy: {best_average_accuracy:.2%}")

    # 使用此最佳 threshold 計算並輸出每份報告書的 accuracy 結果
    summary_results = []
    for report_data in all_reports_data:
        report_name = report_data["report_name"]
        institution = report_data["institution"]
        year = report_data["year"]
        df_all = report_data["df_all"]

        acc = calculate_accuracy_for_report(df_all, best_threshold, RANK_PATH, institution, year, report_name)
        if acc is not None:
            summary_results.append({
                "report_name": report_name,
                "institution": institution,
                "year": year,
                "best_threshold": best_threshold,
                "accuracy_at_best_threshold": acc
            })
            print(f"[INFO] {report_name}: Accuracy at best threshold {best_threshold} = {acc:.2%}")
        else:
            print(f"[WARN] {report_name}: No ground truth available. Skipping summary.")

    summary_df = pd.DataFrame(summary_results)
    summary_path = os.path.join(ACCURACY_RESULT_BASE, "summary_of_all_reports_best_threshold.csv")
    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")
    print(f"[INFO] Summary with best threshold saved to: {summary_path}")

if __name__ == "__main__":
    main()

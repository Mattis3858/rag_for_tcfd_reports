import os
import pandas as pd

# 設定文件路徑
LABEL_RESULT_DIR = "C:\\Users\\bugee\\OneDrive\\桌面\\RAG\\rag_for_tcfd_reports\\data\\label_result"
MERGE_DATA_PATH = "C:\\Users\\bugee\\OneDrive\\桌面\\RAG\\rag_for_tcfd_reports\\data\\merge_data.xlsx"
OUTPUT_DIR = "C:\\Users\\bugee\\OneDrive\\桌面\\RAG\\rag_for_tcfd_reports\\data\\accuracy_result"

# 確保輸出目錄存在
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 定義報告書名稱和對應文件名
report_files = [
    ("2801_彰化銀行_2022_TCFD_報告書", "label_result_2801_彰化銀行_2022_TCFD_報告書.csv"),
    ("2807_渣打銀行_2022_TCFD_報告書", "label_result_2807_渣打銀行_2022_TCFD_報告書.csv"),
    ("2834_台灣企銀_2022_TCFD_報告書", "label_result_2834_台灣企銀_2022_TCFD_報告書.csv"),
    ("2836_高雄銀行_2022_TCFD_報告書", "label_result_2836_高雄銀行_2022_TCFD_報告書.csv"),
    ("2837_凱基銀行_2022_TCFD_報告書", "label_result_2837_凱基銀行_2022_TCFD_報告書.csv")
]

def load_data(label_result_path, merge_data_df, target_filename):
    # 從 target_filename 中提取「銀行 + 年份」
    extracted_filename = target_filename.split('_')[1] + target_filename.split('_')[2][:4]  # 如 "彰化銀行2022"
    
    # 讀取 label_result 和 merge_data 文件
    label_result_df = pd.read_csv(label_result_path)
    target_row = merge_data_df[merge_data_df['filename'] == extracted_filename].iloc[0]
    
    # 提取人工標籤
    true_labels = target_row[target_row == 1].index.tolist()
    
    # 提取模型預測的標籤
    predicted_labels = label_result_df['指標名稱'].unique().tolist()
    
    return true_labels, predicted_labels

def evaluate_predictions(true_labels, predicted_labels):
    # 計算正確、錯誤和遺漏的標籤
    correct_predictions = set(predicted_labels) & set(true_labels)
    incorrect_predictions = set(predicted_labels) - set(true_labels)
    missed_labels = set(true_labels) - set(predicted_labels)
    
    # 準確率
    accuracy = len(correct_predictions) / len(true_labels) if true_labels else 0
    
    # 返回結果
    return {
        "accuracy": accuracy,
        "correct_predictions": list(correct_predictions),
        "incorrect_predictions": list(incorrect_predictions),
        "missed_labels": list(missed_labels)
    }

def main():
    # 載入人工標籤資料
    merge_data_df = pd.read_excel(MERGE_DATA_PATH)
    total_accuracy = 0
    num_reports = len(report_files)

    for report_name, label_result_file in report_files:
        # 加載每份報告書的結果文件
        label_result_path = os.path.join(LABEL_RESULT_DIR, label_result_file)
        true_labels, predicted_labels = load_data(label_result_path, merge_data_df, report_name)
        
        # 計算準確率和分類結果
        result_summary = evaluate_predictions(true_labels, predicted_labels)
        
        # 累加準確率以便計算平均值
        total_accuracy += result_summary['accuracy']
        
        # 打印每份報告的結果
        print(f"報告書: {report_name}")
        print(f"準確率: {result_summary['accuracy'] * 100:.2f}%")
        print("正確分類的標籤:", result_summary['correct_predictions'])
        print("錯誤分類的標籤:", result_summary['incorrect_predictions'])
        print("遺漏的標籤:", result_summary['missed_labels'])
        
        # 將結果存到 CSV
        output_csv_path = os.path.join(OUTPUT_DIR, f"accuracy_result_{report_name}.csv")
        pd.DataFrame([result_summary]).to_csv(output_csv_path, index=False, encoding="utf-8-sig")
        print(f"結果已儲存到：{output_csv_path}\n")

    # 計算並打印平均準確率
    average_accuracy = total_accuracy / num_reports if num_reports > 0 else 0
    print(f"平均準確率: {average_accuracy * 100:.2f}%")

if __name__ == "__main__":
    main()

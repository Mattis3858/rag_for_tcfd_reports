import os
import pandas as pd

# 設定文件路徑
LABEL_RESULT_PATH = "C:\\Users\\bugee\\OneDrive\\桌面\\RAG\\rag_for_tcfd_reports\\data\\label_result\\label_result_彰化銀行2022.csv"
MERGE_DATA_PATH = "C:\\Users\\bugee\\OneDrive\\桌面\\RAG\\rag_for_tcfd_reports\\data\\merge_data.xlsx"
TARGET_FILENAME = "彰化銀行2022"

def load_data():
    # 讀取 label_result 和 merge_data 文件
    label_result_df = pd.read_csv(LABEL_RESULT_PATH)
    merge_data_df = pd.read_excel(MERGE_DATA_PATH)
    return label_result_df, merge_data_df

def extract_labels(label_result_df, merge_data_df, target_filename):
    # 從 merge_data 中提取人工標籤
    target_row = merge_data_df[merge_data_df['filename'] == target_filename].iloc[0]
    true_labels = target_row[target_row == 1].index.tolist()
    
    # 從 label_result 中提取模型預測的標籤
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
    # 載入數據
    label_result_df, merge_data_df = load_data()
    
    # 提取標籤
    true_labels, predicted_labels = extract_labels(label_result_df, merge_data_df, TARGET_FILENAME)
    
    # 計算結果
    result_summary = evaluate_predictions(true_labels, predicted_labels)
    
    # 打印結果
    print(f"準確率: {result_summary['accuracy'] * 100:.2f}%")
    print("正確分類的標籤:", result_summary['correct_predictions'])
    print("錯誤分類的標籤:", result_summary['incorrect_predictions'])
    print("遺漏的標籤:", result_summary['missed_labels'])

if __name__ == "__main__":
    main()

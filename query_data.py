# import argparse
# from transformers import AutoTokenizer, AutoModel
# from langchain_community.vectorstores import Chroma
# from langchain.prompts import ChatPromptTemplate
# import numpy as np

# # Define the path for Chroma database
# CHROMA_PATH = "chroma"

# PROMPT_TEMPLATE = """
# The input text is most similar to the label: {label}
# Based on the following standard:

# {context}

# """

# # Load Hugging Face model and tokenizer for embeddings
# tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")
# model = AutoModel.from_pretrained("bert-base-chinese")

# def generate_embeddings(text: str):
#     inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
#     outputs = model(**inputs)
#     return outputs.last_hidden_state.mean(dim=1).squeeze().tolist()

# # Wrapper class to provide an `embed_query` method for Chroma
# class EmbeddingFunctionWrapper:
#     def embed_query(self, text):
#         return generate_embeddings(text)

# def main():
#     parser = argparse.ArgumentParser()
#     parser.add_argument("query_text", type=str, help="The query text.")
#     args = parser.parse_args()
#     query_text = args.query_text

#     embedding_function = EmbeddingFunctionWrapper()
#     db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

#     # Search the DB with the query
#     results = db.similarity_search_with_relevance_scores(query_text, k=3)

#     # If there are no results, return a message
#     if len(results) == 0:
#         print("Unable to find matching results.")
#         return

#     # Apply score normalization for cosine similarity
#     normalized_results = [
#         (doc, (score + 1) / 2)  # Maps score from [-1, 1] to [0, 1]
#         for doc, score in results
#     ]

#     # Sort and get the highest score result
#     best_result = max(normalized_results, key=lambda x: x[1])
#     best_doc, best_score = best_result

#     # Format and display the result
#     prompt = PROMPT_TEMPLATE.format(label=best_doc.metadata['filename'], context=best_doc.page_content)
#     print(prompt)
#     print(f"Normalized similarity score: {best_score:.2f}")

# if __name__ == "__main__":
#     main()
import argparse
import torch
from transformers import AutoTokenizer, AutoModel
from langchain_community.vectorstores import Chroma
from typing import List

# 定義常數
CHROMA_PATH = "chroma"
MODEL_NAME = "bert-base-chinese"
MAX_LENGTH = 512

class TextEmbedding:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        self.model = AutoModel.from_pretrained(MODEL_NAME)
        self.model.eval()  # 設定模型為評估模式

    def generate_embeddings(self, text: str) -> List[float]:
        """生成文本嵌入向量"""
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=MAX_LENGTH,
            padding=True
        )
        with torch.no_grad():
            outputs = self.model(**inputs)
            embeddings = outputs.last_hidden_state.mean(dim=1).squeeze()
            return embeddings.tolist()

class EmbeddingFunctionWrapper:
    def __init__(self):
        self.text_embedding = TextEmbedding()

    def embed_query(self, text: str) -> List[float]:
        return self.text_embedding.generate_embeddings(text)

class RAGSystem:
    def __init__(self, chroma_path: str):
        self.embedding_function = EmbeddingFunctionWrapper()
        self.db = Chroma(
            persist_directory=chroma_path,
            embedding_function=self.embedding_function
        )

    def search(self, query_text: str, k: int = 5) -> List[dict]:
        """直接使用 Chroma 的相似度分數進行搜索"""
        results = self.db.similarity_search_with_relevance_scores(query_text, k=k)
        return results

def format_results(results: List[dict]) -> str:
    """格式化輸出結果"""
    output = "\n相似度分析結果：\n" + "="*50 + "\n"
    for i, (doc, score) in enumerate(results, 1):
        output += f"\n{i}. 文件標籤: {doc.metadata['filename']}\n"
        output += f"   相似度分數: {score:.4f}\n"
        output += f"   文件內容:\n   {doc.page_content}\n"
        output += "-"*50
    
    # 添加最佳匹配的詳細信息
    best_match = results[0]
    best_doc, best_score = best_match
    output += "\n\n最佳匹配結果：\n"
    output += f"標籤: {best_doc.metadata['filename']}\n"
    output += f"相似度: {best_score:.4f}\n"
    output += f"內容:\n{best_doc.page_content}\n"
    
    return output

def main():
    # 設置命令行參數
    parser = argparse.ArgumentParser(description='RAG System for TCFD Reports')
    parser.add_argument(
        "query_text",
        type=str,
        help="The query text to search for similar documents"
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=5,
        help="Number of top results to return"
    )
    
    args = parser.parse_args()
    
    try:
        # 初始化RAG系統
        rag_system = RAGSystem(CHROMA_PATH)
        
        # 執行搜索
        results = rag_system.search(args.query_text, k=args.top_k)
        
        # 輸出結果
        print(format_results(results))
        
    except Exception as e:
        print(f"錯誤：{str(e)}")
        raise

if __name__ == "__main__":
    main()

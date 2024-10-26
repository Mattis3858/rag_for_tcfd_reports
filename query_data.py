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
from transformers import AutoTokenizer, AutoModel
from langchain_community.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate
import numpy as np
import torch
import torch.nn.functional as F
import warnings
from typing import List, Tuple, Optional
from dataclasses import dataclass

# # 忽略特定警告
# warnings.filterwarnings("ignore", category=UserWarning)

# 定義常數
CHROMA_PATH = "chroma"
MODEL_NAME = "bert-base-chinese"
MAX_LENGTH = 512

@dataclass
class SearchResult:
    label: str
    content: str
    score: float

class TextEmbedding:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        self.model = AutoModel.from_pretrained(MODEL_NAME)
        self.model.eval()  # Set model to evaluation mode

    def generate_embeddings(self, text: str) -> List[float]:
        """生成文本嵌入向量"""
        # 確保文本不為空
        if not text or not text.strip():
            raise ValueError("Input text cannot be empty")

        # Tokenize and encode
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=MAX_LENGTH,
            padding=True
        )

        # Generate embeddings
        with torch.no_grad():
            outputs = self.model(**inputs)
            embeddings = outputs.last_hidden_state.mean(dim=1).squeeze()
            
            # Normalize embeddings
            normalized_embeddings = F.normalize(embeddings, p=2, dim=-1)
            
            return normalized_embeddings.tolist()

class EmbeddingFunctionWrapper:
    def __init__(self):
        self.text_embedding = TextEmbedding()

    def embed_query(self, text: str) -> List[float]:
        return self.text_embedding.generate_embeddings(text)

    def embed_documents(self, documents: List[str]) -> List[List[float]]:
        return [self.text_embedding.generate_embeddings(doc) for doc in documents]

class RAGSystem:
    def __init__(self, chroma_path: str):
        self.embedding_function = EmbeddingFunctionWrapper()
        self.db = Chroma(
            persist_directory=chroma_path,
            embedding_function=self.embedding_function
        )

    @staticmethod
    def compute_similarity(vec1: List[float], vec2: List[float]) -> float:
        """計算兩個向量間的餘弦相似度"""
        vec1 = np.array(vec1)
        vec2 = np.array(vec2)
        
        # 確保向量被歸一化
        vec1 = vec1 / np.linalg.norm(vec1)
        vec2 = vec2 / np.linalg.norm(vec2)
        
        similarity = np.dot(vec1, vec2)
        return float(similarity)

    def search(self, query_text: str, k: int = 5) -> List[SearchResult]:
        """搜索最相關的文檔"""
        # 生成查詢的嵌入向量
        query_embedding = self.embedding_function.embed_query(query_text)
        
        # 搜索相似文檔
        results = self.db.similarity_search_with_relevance_scores(query_text, k=k)
        
        # 處理結果
        search_results = []
        for doc, _ in results:
            doc_embedding = self.embedding_function.embed_query(doc.page_content)
            verified_score = self.compute_similarity(query_embedding, doc_embedding)
            
            search_results.append(SearchResult(
                label=doc.metadata['filename'],
                content=doc.page_content,
                score=verified_score
            ))
        
        # 按相似度分數排序
        search_results.sort(key=lambda x: x.score, reverse=True)
        return search_results

def format_results(results: List[SearchResult]) -> str:
    """格式化輸出結果"""
    output = "\n相似度分析結果：\n" + "="*50 + "\n"
    
    for i, result in enumerate(results, 1):
        output += f"\n{i}. 文件標籤: {result.label}\n"
        output += f"   相似度分數: {result.score:.4f}\n"
        output += f"   文件內容:\n   {result.content}\n"
        output += "-"*50
    
    # 添加最佳匹配的詳細信息
    best_match = results[0]
    output += "\n\n最佳匹配結果：\n"
    output += f"標籤: {best_match.label}\n"
    output += f"相似度: {best_match.score:.4f}\n"
    output += f"內容:\n{best_match.content}\n"
    
    # # 如果最佳匹配的相似度較低，添加建議
    # if best_match.score < 0.95:
    #     output += "\n注意：相似度分數較低，建議：\n"
    #     output += "1. 檢查查詢文本是否完整準確\n"
    #     output += "2. 考慮擴展資料庫中的標準文本\n"
    #     output += "3. 調整匹配算法的參數\n"
    
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
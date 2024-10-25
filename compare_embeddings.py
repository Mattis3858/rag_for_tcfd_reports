from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Load Hugging Face BERT model and tokenizer for embeddings
tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")
model = AutoModel.from_pretrained("bert-base-chinese")

def generate_embeddings(text: str):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).detach().numpy()

def main():
    # Generate embedding for two words
    vector1 = generate_embeddings("apple")
    vector2 = generate_embeddings("iphone")

    # Calculate cosine similarity
    similarity = cosine_similarity(vector1, vector2)[0][0]
    print(f"Cosine similarity between 'apple' and 'iphone': {similarity}")

if __name__ == "__main__":
    main()

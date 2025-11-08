# filepath: d:\WETEC\ai_multi_agents\src\models\embedd.py
import torch
from pathlib import Path
from sentence_transformers import SentenceTransformer
import numpy as np
from src.helppers.helpers import cosine_similarity

class QwenEmbedding:
    """Lớp để tạo embeddings sử dụng Qwen/Qwen3-Embedding-0.6B multilingual model"""
    
    def __init__(self, model_name="Qwen/Qwen3-Embedding-0.6B"):
        print(f"Đang tải {model_name} model...")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = SentenceTransformer(
            model_name,
            device=str(self.device),
            model_kwargs={"torch_dtype": torch.bfloat16} if torch.cuda.is_available() else {}
        )
        print(f"✓ Model {model_name} đã tải xong trên {self.device} với bfloat16")
    
    def get_embedding(self, text):
        """Tạo embedding cho văn bản"""
        if not text or len(text.strip()) == 0:
            return None
        
        # Cắt text nếu quá dài (giới hạn 512 tokens)
        if len(text) > 2000:
            text = text[:2000]
        
        with torch.autocast(device_type=str(self.device).split(':')[0], dtype=torch.bfloat16):
            embedding = self.model.encode(text, convert_to_numpy=True)
        return embedding
    
    def get_embedding_array(self, texts):
        """Tạo ma trận embedding cho danh sách văn bản"""
        if not texts or len(texts) == 0:
            return np.array([])
        
        # Cắt text nếu quá dài (giới hạn 512 tokens)
        processed_texts = []
        for text in texts:
            if len(text) > 2000:
                processed_texts.append(text[:2000])
            else:
                processed_texts.append(text)
        
        with torch.autocast(device_type=str(self.device).split(':')[0], dtype=torch.bfloat16):
            embeddings = self.model.encode(processed_texts, convert_to_numpy=True)
        return embeddings
    
    def calculate_similarity(self, text1, text2) -> float:
        """Tính độ tương đồng giữa 2 văn bản"""
        if not text1 or not text2:
            return 0
        
        emb1 = self.get_embedding(text1)
        emb2 = self.get_embedding(text2)
        
        if emb1 is None or emb2 is None:
            return 0
        
        return cosine_similarity(emb1, emb2)
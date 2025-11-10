import torch
import sys
from pathlib import Path
from sentence_transformers import SentenceTransformer
import numpy as np

# Thêm project root vào sys.path
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.models.helpers import cosine_similarity

class HalongEmbedding:
    """Lớp để tạo embeddings sử dụng hiieu/halong_embedding model cho tiếng Việt"""
    
    def __init__(self, model_name="hiieu/halong_embedding"):
        print(f"Đang tải {model_name} model...")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = SentenceTransformer(
            model_name,
            device=str(self.device),
            model_kwargs={"torch_dtype": torch.bfloat16} if torch.cuda.is_available() else {}
        )
        print(f"✓ Model {model_name} đã tải xong trên {self.device}")
    
    def get_embedding(self, text):
        """
        Tạo embedding cho văn bản
        
        Args:
            text (str): Văn bản cần tạo embedding
            
        Returns:
            numpy.ndarray: Vector embedding
        """
        if not text or len(text.strip()) == 0:
            return None
        
        with torch.autocast(device_type=str(self.device).split(':')[0], dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32):
            embedding = self.model.encode(text, convert_to_numpy=True)
        return embedding
    
    def get_embedding_array(self, texts):
        """
        Tạo ma trận embedding cho danh sách văn bản
        
        Args:
            texts (list[str]): Danh sách văn bản
            
        Returns:
            numpy.ndarray: Ma trận embedding (n_texts, embedding_dim)
        """
        if not texts or len(texts) == 0:
            return np.array([])
        
        # Cắt text nếu quá dài
        processed_texts = []
        for text in texts:
            processed_texts.append(text)
        
        with torch.autocast(device_type=str(self.device).split(':')[0], dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32):
            embeddings = self.model.encode(processed_texts, convert_to_numpy=True)
        return embeddings
    
    def calculate_similarity(self, text1, text2) -> float:
        """
        Tính độ tương đồng cosine giữa 2 văn bản
        
        Args:
            text1 (str): Văn bản thứ nhất
            text2 (str): Văn bản thứ hai
            
        Returns:
            float: Độ tương đồng cosine (0-1)
        """
        if not text1 or not text2:
            return 0.0
        
        emb1 = self.get_embedding(text1)
        emb2 = self.get_embedding(text2)
        
        if emb1 is None or emb2 is None:
            return 0.0
        
        return float(cosine_similarity(emb1, emb2))
    
    def similarity_matrix(self, query_embedding, doc_embeddings):
        """
        Tính ma trận độ tương đồng giữa query và documents
        
        Args:
            query_embedding (numpy.ndarray): Embedding của query (1D hoặc 2D)
            doc_embeddings (numpy.ndarray): Embeddings của documents (2D)
            
        Returns:
            numpy.ndarray: Ma trận độ tương đồng
        """
        # Đảm bảo query_embedding có shape (1, dim)
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)
        
        # Sử dụng model.similarity từ SentenceTransformer
        similarities = self.model.similarity(
            torch.tensor(query_embedding),
            torch.tensor(doc_embeddings)
        )
        
        return similarities.numpy()
    
    def rank_documents(self, query, documents):
        """
        Xếp hạng documents theo độ liên quan với query
        
        Args:
            query (str): Câu truy vấn
            documents (list[str]): Danh sách documents
            
        Returns:
            tuple: (sorted_docs, sorted_scores, sorted_indices)
                - sorted_docs: Documents đã sắp xếp theo độ liên quan
                - sorted_scores: Điểm tương đồng tương ứng
                - sorted_indices: Chỉ số gốc của documents
        """
        if not query or not documents:
            return [], [], []
        
        # Encode query và documents
        query_embedding = self.get_embedding(query)
        doc_embeddings = self.get_embedding_array(documents)
        
        if query_embedding is None or len(doc_embeddings) == 0:
            return [], [], []
        
        # Tính similarity
        similarities = self.similarity_matrix(query_embedding, doc_embeddings).flatten()
        
        # Sort documents by cosine similarity
        sorted_indices = torch.argsort(torch.tensor(similarities), descending=True).numpy()
        sorted_docs = [documents[idx] for idx in sorted_indices]
        sorted_scores = [similarities[idx] for idx in sorted_indices]
        
        return sorted_docs, sorted_scores, sorted_indices
    
    def find_most_similar(self, query, documents, top_k=5):
        """
        Tìm top-k documents tương tự nhất với query
        
        Args:
            query (str): Câu truy vấn
            documents (list[str]): Danh sách documents
            top_k (int): Số lượng kết quả trả về
            
        Returns:
            list[tuple]: Danh sách (document, score, index) được sắp xếp
        """
        sorted_docs, sorted_scores, sorted_indices = self.rank_documents(query, documents)
        
        # Lấy top-k kết quả
        top_k = min(top_k, len(sorted_docs))
        results = [
            (sorted_docs[i], float(sorted_scores[i]), int(sorted_indices[i]))
            for i in range(top_k)
        ]
        
        return results


# Ví dụ sử dụng
if __name__ == "__main__":
    # Khởi tạo model
    embedding = HalongEmbedding()
    
    # Define query and documents
    query = "Bóng đá có lợi ích gì cho sức khỏe?"
    docs = [
        "Bóng đá giúp cải thiện sức khỏe tim mạch và tăng cường sức bền.",
        "Bóng đá là môn thể thao phổ biến nhất thế giới.",
        "Chơi bóng đá giúp giảm căng thẳng và cải thiện tâm lý.",
        "Bóng đá có thể giúp bạn kết nối với nhiều người hơn.",
        "Bóng đá không chỉ là môn thể thao mà còn là cách để giải trí."
    ]
    
    print("=" * 70)
    print("Ví dụ 1: Xếp hạng documents")
    print("=" * 70)
    print(f"Query: {query}\n")
    
    # Xếp hạng documents
    sorted_docs, sorted_scores, sorted_indices = embedding.rank_documents(query, docs)
    
    for doc, score, idx in zip(sorted_docs, sorted_scores, sorted_indices):
        print(f"[{idx}] Score: {score:.4f} - {doc}")
    
    print("\n" + "=" * 70)
    print("Ví dụ 2: Tìm top-3 documents tương tự nhất")
    print("=" * 70)
    
    # Tìm top-3 documents
    top_results = embedding.find_most_similar(query, docs, top_k=3)
    
    for rank, (doc, score, idx) in enumerate(top_results, 1):
        print(f"\nTop {rank}:")
        print(f"  Original Index: {idx}")
        print(f"  Score: {score:.4f}")
        print(f"  Document: {doc}")
    
    print("\n" + "=" * 70)
    print("Ví dụ 3: Tính similarity giữa 2 văn bản")
    print("=" * 70)
    
    text1 = "Bóng đá giúp cải thiện sức khỏe"
    text2 = "Chơi bóng đá tốt cho tim mạch"
    text3 = "Hôm nay trời đẹp quá"
    
    sim_12 = embedding.calculate_similarity(text1, text2)
    sim_13 = embedding.calculate_similarity(text1, text3)
    
    print(f"\nSimilarity('{text1}', '{text2}'): {sim_12:.4f}")
    print(f"Similarity('{text1}', '{text3}'): {sim_13:.4f}")

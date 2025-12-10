from src.models.embedd import QwenEmbedding

# Test CUDA configuration
embedder = QwenEmbedding()
print(f"Device: {embedder.device}")
print(f"CUDA available: {embedder.device.type == 'cuda'}")

# Test basic functionality
text = "Hello world"
emb = embedder.get_embedding(text)
print(f"Embedding shape: {emb.shape if emb is not None else 'None'}")

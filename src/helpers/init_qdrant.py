from qdrant_client import QdrantClient
from langchain_qdrant import QdrantVectorStore
from langchain_openai import OpenAIEmbeddings
from pydantic import SecretStr
import os
from dotenv import load_dotenv
load_dotenv()

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

qdrant_client = QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY,
)


def create_qdrant_vectorstore(collection_name="default_collection", embeddings=None):
    """
    Tạo QdrantVectorStore với cấu hình từ .env

    Args:
        collection_name (str): Tên collection
        embedding_model: Model embedding (mặc định dùng OpenAIEmbeddings)

    Returns:
        QdrantVectorStore: Vector store đã được khởi tạo
    """
    
    model = str(os.getenv("OPENAI_API_MODEL_NAME_EMBED"))
    base_url = os.getenv("OPENAI_BASE_URL_EMBED")
    api_key = str(os.getenv("OPENAI_API_KEY_EMBED"))
    
    if embeddings is None:
        # Sử dụng OpenAI embeddings mặc định (sẽ tự động lấy OPENAI_API_KEY từ env)
        embeddings = OpenAIEmbeddings(
            model=model,
            base_url=base_url,
            api_key=SecretStr(api_key),
            # dimensions=int(os.getenv("EMBED_DIM")),
            tiktoken_enabled=False,
        )

    vectorstore = QdrantVectorStore(
        client=qdrant_client,
        collection_name=collection_name,
        embedding=embeddings
    )

    return vectorstore


def create_vectorstore_from_texts(texts, collection_name="texts_collection", embeddings=None, ids=None, metadatas=None):
    """
    Tạo vector store từ danh sách texts với ids và metadatas tùy chọn

    Args:
        texts (list[str]): Danh sách văn bản
        collection_name (str): Tên collection
        embeddings: Model embedding
        ids (list[str], optional): Danh sách IDs cho từng text
        metadatas (list[dict], optional): Danh sách metadata cho từng text

    Returns:
        QdrantVectorStore: Vector store đã tạo
    """
    if embeddings is None:
        # Sử dụng OpenAI embeddings mặc định (sẽ tự động lấy OPENAI_API_KEY từ env)

        model = str(os.getenv("OPENAI_API_MODEL_NAME_EMBED"))
        base_url = os.getenv("OPENAI_BASE_URL_EMBED")
        api_key = str(os.getenv("OPENAI_API_KEY_EMBED"))

        embeddings = OpenAIEmbeddings(
            model=model,
            base_url=base_url,
            api_key=SecretStr(api_key),
            tiktoken_enabled=False,
        )

    # Tạo vector store với ids và metadatas nếu có
    kwargs = {
        "texts": texts,
        "embedding": embeddings,
        "url": QDRANT_URL,
        "api_key": QDRANT_API_KEY,
        "prefer_grpc": True,
        "collection_name": collection_name
    }

    if ids is not None:
        kwargs["ids"] = ids
    if metadatas is not None:
        kwargs["metadatas"] = metadatas

    vectorstore = QdrantVectorStore.from_texts(**kwargs)

    return vectorstore


# Test connection
if __name__ == "__main__":
    print("🔧 Testing Qdrant connection...")
    try:
        texts = ['text1', 'text2', 'text3']
        vectorstore = create_vectorstore_from_texts(texts, 'texts')
        
        collections = qdrant_client.get_collections()
        print(f"✅ Connected to Qdrant successfully!")
        print(f"📊 Available collections: {[c.name for c in collections.collections]}")

        # # Ví dụ tạo vector store từ texts (commented out vì cần OPENAI_API_KEY)
        # print("\n📝 Note: To create vector stores, add your OPENAI_API_KEY to .env file")
        # print("Example usage:")
        # print("  from src.store.init_qdrant import create_vectorstore_from_texts")
        # print("  texts = ['text1', 'text2', 'text3']")
        # print("  vectorstore = create_vectorstore_from_texts(texts, 'my_collection')")

    except Exception as e:
        print(f"❌ Error connecting to Qdrant: {e}")
        print("Please check your QDRANT_URL and QDRANT_API_KEY in .env file")
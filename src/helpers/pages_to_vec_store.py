import os
import sys
from pathlib import Path
from typing import List, Dict
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from pydantic import SecretStr
import json
import uuid
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add project root to path
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.helpers.init_qdrant import qdrant_client

def read_pages_from_directory(input_dir: str) -> List[Dict[str, str]]:
    """
    Đọc tất cả file txt trong thư mục và tạo list pages (mỗi file là một page)

    Args:
        input_dir (str): Đường dẫn đến thư mục chứa file txt

    Returns:
        List[Dict[str, str]]: List các pages với metadata
    """
    input_path = Path(input_dir)
    pages = []

    # Lấy tất cả file .txt có định dạng page_NUMBER.txt
    txt_files = list(input_path.glob("page_*.txt"))
    
    # Sắp xếp pages theo số thứ tự từ tên file
    def extract_page_num(file_path):
        try:
            return int(file_path.stem.split('_')[1])
        except (IndexError, ValueError):
            return 0
    
    txt_files = sorted(txt_files, key=extract_page_num)

    print(f"📂 Tìm thấy {len(txt_files)} file txt trong {input_dir}")

    for file_path in txt_files:
        file_name = file_path.name
        page_num = extract_page_num(file_path)
        
        if page_num % 20 == 0 or page_num == 1:
            print(f"📄 Đang đọc: {file_name}")

        try:
            # Đọc toàn bộ nội dung file như một page
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read().strip()

            if content:  # Chỉ thêm nếu có nội dung
                # Đếm số paragraphs (tách bởi \n\n)
                paragraphs = [p.strip() for p in content.split("\n\n") if p.strip()]
                
                page = {
                    'page_index': page_num,
                    'seq': len(paragraphs),
                    'content': content,
                    'word_count': len(content.split())
                }
                pages.append(page)

        except Exception as e:
            print(f"   ❌ Lỗi đọc file {file_name}: {e}")

    print(f"✅ Đã đọc thành công {len(pages)} pages")
    return pages

def save_pages_to_json(pages: List[Dict[str, str]], output_file: str):
    """
    Lưu pages vào file JSON

    Args:
        pages (List[Dict[str, str]]): List pages
        output_file (str): Đường dẫn file output
    """
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(pages, f, ensure_ascii=False, indent=2)

    print(f"💾 Đã lưu {len(pages)} pages vào {output_file}")

def store_pages_in_qdrant_direct(pages: List[Dict[str, str]], collection_name: str = "esg_pages") -> QdrantVectorStore:
    """
    Lưu trữ pages trực tiếp vào Qdrant với metadata đầy đủ

    Args:
        pages (List[Dict[str, str]]): List pages với metadata
        collection_name (str): Tên collection trong Qdrant

    Returns:
        QdrantVectorStore: Vector store đã tạo
    """
    from qdrant_client.models import PointStruct, VectorParams, Distance

    print(f"🔧 Đang tạo vector store trực tiếp cho {len(pages)} pages...")

    # Khởi tạo embeddings
    model = str(os.getenv("OPENAI_API_MODEL_NAME_EMBED"))
    base_url = os.getenv("OPENAI_BASE_URL_EMBED")
    api_key = str(os.getenv("OPENAI_API_KEY_EMBED"))

    embeddings = OpenAIEmbeddings(
        model=model,
        base_url=base_url,
        api_key=SecretStr(api_key),
        tiktoken_enabled=False,
    )

    # Xóa collection cũ nếu tồn tại
    try:
        qdrant_client.delete_collection(collection_name)
        print(f"🗑️ Đã xóa collection cũ '{collection_name}'")
    except:
        pass

    # Tạo collection mới với vector size phù hợp
    sample_embedding = embeddings.embed_query("test")
    vector_size = len(sample_embedding)

    qdrant_client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE)
    )
    print(f"📁 Đã tạo collection mới '{collection_name}' với vector size {vector_size}")

    # Tạo points với payload đầy đủ
    points = []
    for i, page in enumerate(pages):
        if i % 10 == 0:  # Hiển thị tiến trình mỗi 10 pages
            print(f"📊 Đang xử lý page {i+1}/{len(pages)}...")

        # Tạo embedding cho toàn bộ nội dung page
        vector = embeddings.embed_query(page['content'])

        # Tạo UUID cho point ID
        point_id = str(uuid.uuid4())

        # Tạo payload với tất cả metadata
        payload = {
            'page_index': page['page_index'],
            'content': page['content'],
            'seq': page['seq'],
            'word_count': page['word_count'],
            'page_content': page['content']  # Để tương thích với LangChain
        }

        # Tạo point
        point = PointStruct(
            id=point_id,  # Sử dụng UUID làm ID
            vector=vector,
            payload=payload
        )
        points.append(point)

    # Upload points theo batch
    batch_size = 50  # Batch nhỏ hơn vì mỗi page có thể lớn
    for i in range(0, len(points), batch_size):
        batch = points[i:i+batch_size]
        qdrant_client.upsert(
            collection_name=collection_name,
            points=batch
        )
        print(f"📤 Đã upload batch {i//batch_size + 1}/{(len(points)-1)//batch_size + 1}")

    print(f"✅ Đã tạo vector store '{collection_name}' với {len(points)} pages")

    # Tạo QdrantVectorStore wrapper để sử dụng với LangChain
    vectorstore = QdrantVectorStore.from_existing_collection(
        embedding=embeddings,
        collection_name=collection_name,
        url=os.getenv("QDRANT_URL"),
        api_key=os.getenv("QDRANT_API_KEY"),
        prefer_grpc=True
    )

    return vectorstore

def retrieve_similar_pages(query: str, vectorstore, top_k: int = 5):
    """
    Từ câu hỏi, retrieval các pages có nội dung tương đồng nhất

    Args:
        query (str): Câu hỏi cần tìm
        vectorstore: QdrantVectorStore đã được tạo
        top_k (int): Số lượng kết quả trả về

    Returns:
        List[Dict]: Danh sách kết quả với id, content, và score
    """
    print(f"🔍 Đang tìm kiếm cho query: '{query}'")

    # Khởi tạo embeddings
    model = str(os.getenv("OPENAI_API_MODEL_NAME_EMBED"))
    base_url = os.getenv("OPENAI_BASE_URL_EMBED")
    api_key = str(os.getenv("OPENAI_API_KEY_EMBED"))

    embeddings = OpenAIEmbeddings(
        model=model,
        base_url=base_url,
        api_key=SecretStr(api_key),
        tiktoken_enabled=False,
    )

    # Tạo embedding cho query
    query_vector = embeddings.embed_query(query)

    # Tìm kiếm trực tiếp từ Qdrant
    search_results = qdrant_client.search(
        collection_name="esg_pages",
        query_vector=query_vector,
        limit=top_k,
        with_payload=True,
        with_vectors=False
    )

    # Format kết quả
    formatted_results = []
    for result in search_results:
        payload = result.payload or {}
        formatted_results.append({
            'page_index': payload.get('page_index', 'N/A'),
            'content': payload.get('content', ''),
            'seq': payload.get('seq', 'N/A'),
            'word_count': payload.get('word_count', 'N/A'),
            'similarity_score': float(result.score)
        })

    return formatted_results

def display_page_retrieval_results(results: List[Dict]):
    """
    Hiển thị kết quả retrieval cho pages

    Args:
        results (List[Dict]): Kết quả từ retrieve_similar_pages
    """
    print(f"\n📋 KẾT QUẢ RETRIEVAL top-k: {len(results)}")
    print("=" * 10)

    for i, result in enumerate(results, 1):
        print(f"\n{i}. [SEARCH] Ở Page: {result['page_index']}")
        print(f"   Sequences: {result['seq']}")
        print(f"   Words: {result['word_count']}")
        print(f"   Similarity Score: {result['similarity_score']:.4f}")
        print(f"   Content: \n{result['content'][:200]}{'...' if len(result['content']) > 200 else ''}")

    print("\n" + "=" * 80)

# ========== PIPELINE ==========

if __name__ == "__main__":
    # Cấu hình đường dẫn
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent.parent
    input_dir = project_root / "src" / "data" / "results" / "grammar"
    output_dir = project_root / "src" / "store" / "data_to_push"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("🚀 BẮT ĐẦU XỬ LÝ PAGES")
    print(f"📂 Input:  {input_dir}")
    print(f"📂 Output: {output_dir}")

    # Đọc và xử lý pages
    pages = read_pages_from_directory(str(input_dir))

    # Hiển thị thống kê
    print(f"\n📊 THỐNG KÊ:")
    print(f"   Tổng số pages: {len(pages)}")
    total_words = sum(int(page['word_count']) for page in pages)
    print(f"   Tổng từ: {total_words:,}")

    # Thống kê chi tiết
    print(f"\n📋 CHI TIẾT THEO PAGE:")
    for page in pages:
        print(f"   Page {page['page_index']}: {page['word_count']:,} words, {page['seq']} sequences")

    # Lưu kết quả JSON
    output_file = output_dir / "pages_data.json"
    save_pages_to_json(pages, str(output_file))

    # Lưu vào Qdrant
    try:
        vectorstore = store_pages_in_qdrant_direct(pages, "esg_pages")
        print(f"\n✅ Đã lưu trữ thành công vào Qdrant!")

        # Test retrieval
        print(f"\n🧪 TEST RETRIEVAL:")
        test_queries = [
            "thể dục thể thao sức khỏe dồi dào",
            "Vốn điều lệ",
            "quản lý rủi ro",
            "doanh thu năm 2024"
        ]

        for query in test_queries:
            results = retrieve_similar_pages(query, vectorstore, top_k=3)
            display_page_retrieval_results(results)
            print("\n" + "-"*50)

    except Exception as e:
        print(f"\n❌ Lỗi lưu trữ vào Qdrant: {e}")
        print("Vui lòng kiểm tra OPENAI_API_KEY_EMBED trong .env file")
    
    # Hiển thị vài examples
    print(f"\n📝 VÍ DỤ PAGES:")
    for i, page in enumerate(pages[:3], 1):
        print(f"\n{i}. Ở trang {page['page_index']}")
        print(f"   Words: {page['word_count']:,}")
        print(f"   Sequences: {page['seq']}")
        print(f"   Content: \n{page['content'][:150]}{'...' if len(page['content']) > 150 else ''}")

    print(f"\n✅ HOÀN THÀNH! Đã xử lý {len(pages)} pages từ thư mục {input_dir}")
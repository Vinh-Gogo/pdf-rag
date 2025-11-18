import os
import sys
from pathlib import Path
from typing import List, Dict, Optional
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from pydantic import SecretStr
import json
import uuid
from dotenv import load_dotenv
import re
from rank_bm25 import BM25Okapi

# Load environment variables
load_dotenv()

# Add project root to path
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.helpers.init_qdrant import qdrant_client


# ============================================================================
# BM25 SUPPORT for Pages (sparse lexical search)
# ============================================================================

_bm25_pages_index: Optional[BM25Okapi] = None
_bm25_pages_tokens_corpus: Optional[List[List[str]]] = None
_bm25_pages: Optional[List[Dict]] = None


def _simple_tokenize(text: str) -> List[str]:
    return re.findall(r"\w+", text.lower(), flags=re.UNICODE)


def build_bm25_pages_index(pages: List[Dict]) -> None:
    global _bm25_pages_index, _bm25_pages_tokens_corpus, _bm25_pages
    _bm25_pages = pages
    _bm25_pages_tokens_corpus = [_simple_tokenize(p.get("content", "")) for p in pages]
    _bm25_pages_index = BM25Okapi(_bm25_pages_tokens_corpus)


def ensure_bm25_pages_index_initialized(pages: Optional[List[Dict]] = None) -> None:
    global _bm25_pages_index
    if _bm25_pages_index is None:
        if pages is None:
            raise RuntimeError("BM25 pages index is not initialized. Call build_bm25_pages_index(pages) first or run the pipeline.")
        build_bm25_pages_index(pages)

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
    txt_files = list(input_path.glob("page_cleared_*.txt"))
    
    # Sắp xếp pages theo số thứ tự từ tên file
    def extract_page_num(file_path):
        try:
            return int(file_path.stem.split('_')[2])
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
                # Đếm số paragraphs (tách bởi #)
                paragraphs = [p.strip() for p in content.split("#") if p.strip()]
                
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
            'page_id': f"page_{page['page_index']}",
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
    batch_size = 20  # Batch nhỏ hơn vì mỗi page có thể lớn
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

def retrieve_similar_pages(query: str, vectorstore=None, top_k: int = 5, collection_name: str = "esg_pages"):
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
        collection_name=collection_name,
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
            'page_id': payload.get('page_id', f"page_{payload.get('page_index', 'N/A')}") or f"page_{payload.get('page_index', 'N/A')}",
            'page_index': payload.get('page_index', 'N/A'),
            'content': payload.get('content', ''),
            'seq': payload.get('seq', 'N/A'),
            'word_count': payload.get('word_count', 'N/A'),
            'similarity_score': float(result.score)
        })

    return formatted_results

def display_page_results(results: List[Dict]):
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
        base = f"   Words: {result['word_count']} | Score: {result['similarity_score']:.4f}"
        emb = result.get('embedding_score')
        bm25 = result.get('bm25_score')
        if emb is not None or bm25 is not None:
            parts = []
            if emb is not None:
                parts.append(f"emb={emb:.4f}")
            if bm25 is not None:
                parts.append(f"bm25={bm25:.4f}")
            base += " (" + ", ".join(parts) + ")"
        print(base)
        print(f"   Content: \n{result['content'][:200]}{'...' if len(result['content']) > 200 else ''}")
        
def retrieve_bm25_pages(query: str, top_k: int = 5, pages: Optional[List[Dict]] = None) -> List[Dict]:
    """
    Retrieve top-k pages using BM25 lexical search.
    """
    ensure_bm25_pages_index_initialized(pages)
    assert _bm25_pages_index is not None and _bm25_pages is not None

    query_tokens = _simple_tokenize(query)
    scores = _bm25_pages_index.get_scores(query_tokens)
    ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]

    results: List[Dict] = []
    for i in ranked_indices:
        page = _bm25_pages[i]
        results.append({
            'page_id': f"page_{page.get('page_index', 'N/A')}",
            'page_index': page.get('page_index', 'N/A'),
            'content': page.get('content', ''),
            'seq': page.get('seq', 'N/A'),
            'word_count': page.get('word_count', 'N/A'),
            'similarity_score': float(scores[i]),
            'bm25_score': float(scores[i])
        })
    return results


def _min_max_norm(scores: Dict[str, float]) -> Dict[str, float]:
    if not scores:
        return {}
    vals = list(scores.values())
    mn, mx = min(vals), max(vals)
    if mx - mn == 0:
        return {k: 0.0 for k in scores}
    return {k: (v - mn) / (mx - mn) for k, v in scores.items()}


def retrieve_hybrid_pages(
    query: str,
    top_k: int = 5,
    alpha: float = 0.5,
    pages: Optional[List[Dict]] = None,
    collection_name: str = "esg_pages"
) -> List[Dict]:
    """
    Hybrid retrieval combining dense (Qdrant) and sparse (BM25) for pages.

    combined = alpha * dense_norm + (1 - alpha) * bm25_norm
    """
    ensure_bm25_pages_index_initialized(pages)

    # Dense top candidates
    dense_results = retrieve_similar_pages(query, vectorstore=None, top_k=max(top_k, 10), collection_name=collection_name)
    dense_scores = {r.get('page_id'): float(r['similarity_score']) for r in dense_results}

    # BM25 broader set
    bm25_results = retrieve_bm25_pages(query, top_k=max(top_k, 50))
    bm25_scores = {r.get('page_id'): float(r['bm25_score']) for r in bm25_results}

    candidate_ids = set(dense_scores) | set(bm25_scores)
    dense_norm = _min_max_norm({k: dense_scores.get(k, 0.0) for k in candidate_ids})
    bm25_norm = _min_max_norm({k: bm25_scores.get(k, 0.0) for k in candidate_ids})

    combined = {k: alpha * dense_norm.get(k, 0.0) + (1 - alpha) * bm25_norm.get(k, 0.0) for k in candidate_ids}

    # Build lookup map from pages corpus
    page_lookup: Dict[str, Dict] = {}
    if _bm25_pages is not None:
        for p in _bm25_pages:
            page_lookup[f"page_{p.get('page_index', 'N/A')}"] = p

    ranked_ids = sorted(candidate_ids, key=lambda k: combined[k], reverse=True)[:top_k]
    out: List[Dict] = []
    for pid in ranked_ids:
        p = page_lookup.get(pid, {})
        out.append({
            'page_id': pid,
            'page_index': p.get('page_index', 'N/A'),
            'content': p.get('content', ''),
            'seq': p.get('seq', 'N/A'),
            'word_count': p.get('word_count', 'N/A'),
            'similarity_score': float(combined[pid]),
            'embedding_score': float(dense_scores.get(pid, 0.0)),
            'bm25_score': float(bm25_scores.get(pid, 0.0)),
        })
    return out

    print("\n" + "=" * 80)

# ========== PIPELINE ==========

if __name__ == "__main__":
    # Cấu hình đường dẫn
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent.parent
    input_dir = project_root / "src" / "data" / "md_to_plain_text"
    output_dir = project_root / "src" / "data" / "push"
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

    # Xây dựng chỉ mục BM25 cho pages
    try:
        build_bm25_pages_index(pages)
        print(f"\n🔎 BM25: đã xây dựng chỉ mục lexical cho {len(pages):,} pages")
    except Exception as e:
        print(f"❌ Lỗi khi xây dựng BM25 index cho pages: {e}")

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
            results = retrieve_similar_pages(query, vectorstore, top_k=3, collection_name="esg_pages")
            display_page_results(results)

            print("\n[BM25 - Lexical]")
            bm25_results = retrieve_bm25_pages(query, top_k=3, pages=pages)
            display_page_results(bm25_results)

            print("\n[Hybrid - Combined]")
            hybrid_results = retrieve_hybrid_pages(query, top_k=3, alpha=0.6, pages=pages, collection_name="esg_pages")
            display_page_results(hybrid_results)
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
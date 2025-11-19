"""
Sequences to Vector Store Pipeline

Quy trình:
1. Đọc tất cả file text từ source data
2. Tách mỗi file thành các sequences (đoạn văn) (Lượt bỏ các đoạn quá ngắn)
3. Lưu tất cả sequences vào Qdrant Vector Store với metadata
4. Test retrieval
"""

import os
import sys
from pathlib import Path
from typing import List, Dict, Optional
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from pydantic import SecretStr
import json
import uuid
import re
from rank_bm25 import BM25Okapi

# Add project root to path
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.helpers.init_qdrant import qdrant_client

# ============================================================================
# BM25 SUPPORT (sparse lexical search)
# ============================================================================

_bm25_index: Optional[BM25Okapi] = None
_bm25_tokens_corpus: Optional[List[List[str]]] = None
_bm25_sequences: Optional[List[Dict]] = None  # keep same objects as input


def _simple_tokenize(text: str) -> List[str]:
    # Basic unicode-aware tokenization; good enough for VN and EN
    return re.findall(r"\w+", text.lower(), flags=re.UNICODE)


def build_bm25_index(sequences: List[Dict]) -> None:
    """
    Build a BM25 index from sequences' content for lexical retrieval.

    Args:
        sequences (List[Dict]): sequences as produced by read_sequences_from_directory
    """
    global _bm25_index, _bm25_tokens_corpus, _bm25_sequences
    _bm25_sequences = sequences
    _bm25_tokens_corpus = [_simple_tokenize(s.get("content", "")) for s in sequences]
    _bm25_index = BM25Okapi(_bm25_tokens_corpus)


def ensure_bm25_index_initialized(sequences: Optional[List[Dict]] = None) -> None:
    """
    Ensure BM25 is ready. If not, and sequences provided, build it.
    """
    global _bm25_index
    if _bm25_index is None:
        if sequences is None:
            raise RuntimeError("BM25 index is not initialized. Call build_bm25_index(sequences) first or run the pipeline.")
        build_bm25_index(sequences)


def extract_page_number(filename: str) -> int:
    """
    Trích xuất số trang từ tên file (page_1.txt -> 1)
    
    Args:
        filename (str): Tên file
        
    Returns:
        int: Số trang
    """
    try:
        match = re.search(r'page_cleared_(\d+)', filename)
        if match:
            return int(match.group(1))
        return 0
    except:
        return 0


def read_sequences_from_directory(input_dir: str, min_words: int = 10) -> List[Dict]:
    """
    Đọc tất cả file txt và tách thành sequences (đoạn văn)
    
    Args:
        input_dir (str): Đường dẫn đến thư mục chứa file txt
        min_words (int): Số từ tối thiểu cho một sequence
        
    Returns:
        List[Dict]: List các sequences với metadata
    """
    input_path = Path(input_dir)
    all_sequences = []
    
    # Lấy tất cả file .txt có định dạng page_NUMBER.txt
    txt_files = list(input_path.glob("page_cleared_*.txt"))
    
    # Sắp xếp theo số trang
    txt_files = sorted(txt_files, key=lambda x: extract_page_number(x.name))
    
    print(f"📂 Tìm thấy {len(txt_files)} file txt trong {input_dir}")
    print(f"🔍 Sẽ tách các sequences với tối thiểu {min_words} từ\n")
    
    total_sequences = 0
    
    for file_path in txt_files:
        page_num = extract_page_number(file_path.name)
        
        try:
            # Đọc nội dung file
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read().strip()
            
            if not content:
                continue
            
            # Tách thành sequences bằng \n\n (double newline)
            raw_sequences = [seq for seq in content.split('#') if seq]
            
            # Filter và clean sequences
            page_sequences = []
            for seq_idx, seq_text in enumerate(raw_sequences, 1):
                seq_text = seq_text.strip()
                
                if not seq_text:
                    continue
                
                # Đếm số từ
                word_count = len(seq_text.split())
                
                # Chỉ giữ sequences có đủ số từ
                if word_count >= min_words:
                    sequence = {
                        'page_index': page_num,
                        'seq_index': seq_idx,
                        'seq_id': f"page_{page_num}_seq_{seq_idx}",
                        'content': seq_text,
                        'word_count': word_count,
                        'char_count': len(seq_text)
                    }
                    page_sequences.append(sequence)
            
            all_sequences.extend(page_sequences)
            total_sequences += len(page_sequences)
            
            if page_num % 20 == 0 or page_num == 1:
                print(f"📄 Page {page_num}: {len(page_sequences)} sequences")
        
        except Exception as e:
            print(f"❌ Lỗi đọc file {file_path.name}: {e}")
    
    print(f"\n✅ Đã tách thành công {total_sequences} sequences từ {len(txt_files)} pages")
    return all_sequences


def save_sequences_to_json(sequences: List[Dict], output_file: str):
    """
    Lưu sequences vào file JSON
    
    Args:
        sequences (List[Dict]): List sequences
        output_file (str): Đường dẫn file output
    """
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(sequences, f, ensure_ascii=False, indent=2)
    
    print(f"💾 Đã lưu {len(sequences)} sequences vào {output_file}")


def store_sequences_in_qdrant(
    sequences: List[Dict], 
    collection_name: str = "esg_sequences",
    batch_size: int = 50
) -> QdrantVectorStore:
    """
    Lưu trữ sequences vào Qdrant với metadata đầy đủ
    
    Args:
        sequences (List[Dict]): List sequences với metadata
        collection_name (str): Tên collection trong Qdrant
        batch_size (int): Số sequences mỗi batch
        
    Returns:
        QdrantVectorStore: Vector store đã tạo
    """
    from qdrant_client.models import PointStruct, VectorParams, Distance
    
    print(f"\n🔧 Đang tạo vector store cho {len(sequences)} sequences...")
    
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
    
    # Tạo collection mới
    sample_embedding = embeddings.embed_query("test")
    vector_size = len(sample_embedding)
    
    qdrant_client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE)
    )
    print(f"📁 Đã tạo collection '{collection_name}' với vector size {vector_size}")
    
    # Tạo points với embedding
    print(f"\n📊 Đang tạo embeddings cho {len(sequences)} sequences...")
    points = []
    
    for i, seq in enumerate(sequences):
        # Hiển thị tiến trình
        if (i + 1) % 20 == 0 or i == 0:
            print(f"   🔄 Processing: {i+1}/{len(sequences)} sequences...")
        
        # Tạo embedding cho sequence
        vector = embeddings.embed_query(seq['content'])
        
        # Tạo point
        point = PointStruct(
            id=str(uuid.uuid4()),
            vector=vector,
            payload={
                'page_index': seq['page_index'],
                'seq_index': seq['seq_index'],
                'seq_id': seq['seq_id'],
                'content': seq['content'],
                'word_count': seq['word_count'],
                'char_count': seq['char_count'],
                'page_content': seq['content']  # Để tương thích với LangChain
            }
        )
        points.append(point)
    
    # Upload points theo batch
    print(f"\n📤 Đang upload {len(points)} points vào Qdrant...")
    total_batches = (len(points) - 1) // batch_size + 1
    
    for i in range(0, len(points), batch_size):
        batch = points[i:i+batch_size]
        qdrant_client.upsert(
            collection_name=collection_name,
            points=batch
        )
        batch_num = i // batch_size + 1
        print(f"   ✅ Uploaded batch {batch_num}/{total_batches}")
    
    print(f"\n✅ Đã tạo vector store '{collection_name}' với {len(points)} sequences")
    
    # Tạo QdrantVectorStore wrapper
    vectorstore = QdrantVectorStore.from_existing_collection(
        embedding=embeddings,
        collection_name=collection_name,
        url=os.getenv("QDRANT_URL"),
        api_key=os.getenv("QDRANT_API_KEY"),
        prefer_grpc=True
    )
    
    return vectorstore


def retrieve_similar_sequences(
    query: str, 
    collection_name: str = "esg_sequences",
    top_k: int = 5
) -> List[Dict]:
    """
    Tìm kiếm sequences tương tự với query
    
    Args:
        query (str): Câu hỏi cần tìm
        collection_name (str): Tên collection
        top_k (int): Số lượng kết quả
        
    Returns:
        List[Dict]: Danh sách kết quả
    """
    print(f"\n🔍 Đang tìm kiếm: '{query}'")
    
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
    
    # Tìm kiếm
    search_results = qdrant_client.search(
        collection_name=collection_name,
        query_vector=query_vector,
        limit=top_k,
        with_payload=True,
        with_vectors=False
    )
    
    # Format kết quả
    results = []
    for result in search_results:
        payload = result.payload or {}
        results.append({
            'seq_id': payload.get('seq_id', 'N/A'),
            'page_index': payload.get('page_index', 'N/A'),
            'seq_index': payload.get('seq_index', 'N/A'),
            'content': payload.get('content', ''),
            'word_count': payload.get('word_count', 'N/A'),
            'similarity_score': float(result.score)
        })
    
    return results


def retrieve_bm25_sequences(
    query: str,
    top_k: int = 5,
    sequences: Optional[List[Dict]] = None
) -> List[Dict]:
    """
    Retrieve similar sequences using BM25 lexical search over in-memory corpus.

    Args:
        query (str): query text
        top_k (int): number of results
        sequences (Optional[List[Dict]]): if provided, (re)builds the BM25 index from this list

    Returns:
        List[Dict]: results formatted similarly to embedding retrieval
    """
    ensure_bm25_index_initialized(sequences)

    assert _bm25_index is not None and _bm25_sequences is not None
    query_tokens = _simple_tokenize(query)
    scores = _bm25_index.get_scores(query_tokens)

    # Get top_k indices
    ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]

    results = []
    for i in ranked_indices:
        seq = _bm25_sequences[i]
        results.append({
            'seq_id': seq.get('seq_id', 'N/A'),
            'page_index': seq.get('page_index', 'N/A'),
            'seq_index': seq.get('seq_index', 'N/A'),
            'content': seq.get('content', ''),
            'word_count': seq.get('word_count', 'N/A'),
            'similarity_score': float(scores[i]),  # keep key for display compatibility
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


def retrieve_hybrid_sequences(
    query: str,
    collection_name: str = "esg_sequences",
    top_k: int = 5,
    alpha: float = 0.5,
    sequences: Optional[List[Dict]] = None
) -> List[Dict]:
    """
    Hybrid retrieval by combining dense (Qdrant embedding) and sparse (BM25) signals.

    combined_score = alpha * dense_norm + (1 - alpha) * bm25_norm

    Args:
        query (str): query text
        collection_name (str): qdrant collection for dense search
        top_k (int): number of final results
        alpha (float): weight for dense score in [0,1]
        sequences (Optional[List[Dict]]): optional corpus to (re)build BM25 index
    """
    # Ensure BM25
    ensure_bm25_index_initialized(sequences)

    # Dense results
    dense_results = retrieve_similar_sequences(query, collection_name=collection_name, top_k=max(top_k, 10))
    dense_scores = {r['seq_id']: float(r['similarity_score']) for r in dense_results}

    # BM25 results over the whole corpus
    bm25_results = retrieve_bm25_sequences(query, top_k=max(top_k, 50))
    bm25_scores = {r['seq_id']: float(r['bm25_score']) for r in bm25_results}

    # Union candidate set
    candidate_ids = set(dense_scores) | set(bm25_scores)

    # Normalize
    dense_norm = _min_max_norm({k: dense_scores.get(k, 0.0) for k in candidate_ids})
    bm25_norm = _min_max_norm({k: bm25_scores.get(k, 0.0) for k in candidate_ids})

    combined = {k: alpha * dense_norm.get(k, 0.0) + (1 - alpha) * bm25_norm.get(k, 0.0) for k in candidate_ids}

    # Build lookup map for sequence details (prefer from BM25 corpus, fallback to dense payloads)
    seq_lookup: Dict[str, Dict] = {}
    if _bm25_sequences is not None:
        for s in _bm25_sequences:
            seq_lookup[s.get('seq_id', '')] = s

    # Also enrich from dense results payload if needed
    for r in dense_results:
        seq_id = r.get('seq_id')
        if seq_id and seq_id not in seq_lookup:
            seq_lookup[seq_id] = {
                'seq_id': seq_id,
                'page_index': r.get('page_index'),
                'seq_index': r.get('seq_index'),
                'content': r.get('content'),
                'word_count': r.get('word_count'),
            }

    # Rank and format
    ranked_ids = sorted(candidate_ids, key=lambda k: combined[k], reverse=True)[:top_k]
    out: List[Dict] = []
    for sid in ranked_ids:
        s = seq_lookup.get(sid, {})
        out.append({
            'seq_id': sid,
            'page_index': s.get('page_index', 'N/A'),
            'seq_index': s.get('seq_index', 'N/A'),
            'content': s.get('content', ''),
            'word_count': s.get('word_count', 'N/A'),
            'similarity_score': float(combined[sid]),
            'embedding_score': float(dense_scores.get(sid, 0.0)),
            'bm25_score': float(bm25_scores.get(sid, 0.0)),
        })

    return out


def display_sequences_results(results: List[Dict]):
    """
    Hiển thị kết quả retrieval
    
    Args:
        results (List[Dict]): Kết quả từ retrieve_similar_sequences
    """
    print(f"\n📋 KẾT QUẢ RETRIEVAL (Top-{len(results)}):")
    print("=" * 80)
    
    for i, result in enumerate(results, 1):
        print(f"\n{i}. [{result['seq_id']}]")
        print(f"   📄 Page: {result['page_index']} | Sequence: {result['seq_index']}")
        base = f"   📊 Words: {result['word_count']} | Score: {result['similarity_score']:.4f}"
        # Show optional component scores if present
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
        print(f"   📝 Content:")
        
        # Hiển thị content với wrap
        content = result['content']
        if len(content) > 200:
            print(f"      {content[:200]}...")
        else:
            print(f"      {content}")
    
    print("\n" + "=" * 80)


def display_statistics(sequences: List[Dict]):
    """
    Hiển thị thống kê về sequences
    
    Args:
        sequences (List[Dict]): List sequences
    """
    if not sequences:
        return
    
    total_sequences = len(sequences)
    total_words = sum(s['word_count'] for s in sequences)
    total_chars = sum(s['char_count'] for s in sequences)
    
    # Thống kê theo page
    pages = {}
    for seq in sequences:
        page = seq['page_index']
        if page not in pages:
            pages[page] = {'count': 0, 'words': 0}
        pages[page]['count'] += 1
        pages[page]['words'] += seq['word_count']
    
    print(f"\n{'='*80}")
    print("📊 THỐNG KÊ SEQUENCES:")
    print("="*80)
    print(f"📝 Tổng sequences: {total_sequences:,}")
    print(f"📄 Tổng pages: {len(pages):,}")
    print(f"💬 Tổng từ: {total_words:,}")
    print(f"📊 Trung bình: {total_words/total_sequences:.1f} từ/sequence")
    print(f"📏 Tổng ký tự: {total_chars:,}")
    
    # Top pages có nhiều sequences nhất
    top_pages = sorted(pages.items(), key=lambda x: x[1]['count'], reverse=True)[:5]
    print(f"\n🔝 Top 5 pages có nhiều sequences nhất:")
    for page_num, stats in top_pages:
        print(f"   Page {page_num}: {stats['count']} sequences, {stats['words']:,} words")
    
    print("="*80)


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def run_sequences_pipeline(
    input_dir: str,
    output_json: str,
    collection_name: str = "esg_sequences",
    min_words: int = 10,
    batch_size: int = 50,
    skip_json: bool = False,
    skip_vectorstore: bool = False
):
    """
    Chạy pipeline hoàn chỉnh: đọc sequences -> lưu JSON -> lưu vector store
    
    Args:
        input_dir (str): Thư mục chứa file txt
        output_json (str): File JSON output
        collection_name (str): Tên collection Qdrant
        min_words (int): Số từ tối thiểu cho sequence
        batch_size (int): Batch size khi upload
        skip_json (bool): Bỏ qua lưu JSON
        skip_vectorstore (bool): Bỏ qua lưu vector store
    """
    print("="*80)
    print("🚀 SEQUENCES TO VECTOR STORE PIPELINE")
    print("="*80)
    print(f"📂 Input dir: {input_dir}")
    print(f"📄 Output JSON: {output_json}")
    print(f"🗄️ Collection: {collection_name}")
    print(f"🔢 Min words: {min_words}")
    print("="*80)
    
    # Step 1: Đọc và tách sequences
    print(f"\n{'='*80}")
    print("BƯỚC 1: ĐỌC VÀ TÁCH SEQUENCES")
    print("="*80)
    
    sequences = read_sequences_from_directory(input_dir, min_words=min_words)
    
    if not sequences:
        print("❌ Không tìm thấy sequences nào!")
        return False
    
    # Hiển thị thống kê
    display_statistics(sequences)

    # Xây dựng chỉ mục BM25 cho tìm kiếm lexical
    try:
        build_bm25_index(sequences)
        print(f"\n🔎 BM25: đã xây dựng chỉ mục lexical cho {len(sequences):,} sequences")
    except Exception as e:
        print(f"❌ Lỗi khi xây dựng BM25 index: {e}")
    
    # Step 2: Lưu JSON
    if not skip_json:
        print(f"\n{'='*80}")
        print("BƯỚC 2: LƯU SEQUENCES VÀO JSON")
        print("="*80)
        save_sequences_to_json(sequences, output_json)
    else:
        print(f"\n⏭️ Bỏ qua BƯỚC 2: Lưu JSON (skip_json=True)")
    
    # Step 3: Lưu vào Vector Store
    if not skip_vectorstore:
        print(f"\n{'='*80}")
        print("BƯỚC 3: LƯU VÀO QDRANT VECTOR STORE")
        print("="*80)
        
        try:
            vectorstore = store_sequences_in_qdrant(
                sequences, 
                collection_name=collection_name,
                batch_size=batch_size
            )
            
            # Step 4: Test retrieval
            print(f"\n{'='*80}")
            print("BƯỚC 4: TEST RETRIEVAL")
            print("="*80)
            
            test_queries = [
                "vốn điều lệ của công ty",
                "báo cáo tài chính năm 2024",
                "hoạt động kinh doanh chính",
                "quản trị rủi ro và tuân thủ",
                "phát triển bền vững ESG"
            ]
            
            for query in test_queries:
                print(f"\n--- Query: {query} ---")

                print("\n[Dense Embedding - Qdrant]")
                dense_results = retrieve_similar_sequences(
                    query,
                    collection_name=collection_name,
                    top_k=3
                )
                display_sequences_results(dense_results)

                print("\n[BM25 - Lexical]")
                bm25_results = retrieve_bm25_sequences(query, top_k=3)
                display_sequences_results(bm25_results)

                print("\n[Hybrid - Combined]")
                hybrid_results = retrieve_hybrid_sequences(
                    query,
                    collection_name=collection_name,
                    top_k=3,
                    alpha=0.6  # favor embeddings slightly
                )
                display_sequences_results(hybrid_results)
            
        except Exception as e:
            print(f"❌ Lỗi khi lưu vào vector store: {e}")
            import traceback
            traceback.print_exc()
            return False
    else:
        print(f"\n⏭️ Bỏ qua BƯỚC 3-4: Vector store (skip_vectorstore=True)")
    
    # Final summary
    print(f"\n{'='*80}")
    print("✅ PIPELINE HOÀN THÀNH!")
    print("="*80)
    print(f"📝 Tổng sequences: {len(sequences):,}")
    if not skip_json:
        print(f"📄 JSON saved: {output_json}")
    if not skip_vectorstore:
        print(f"🗄️ Vector store: {collection_name}")
    print("="*80)
    
    return True


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    # Load environment variables
    from dotenv import load_dotenv
    load_dotenv()
    
    # Cấu hình
    INPUT_DIR = project_root / "src" / "data" / "contents" / "drop"
    OUTPUT_JSON = project_root / "src" / "data" / "push" / "sequences" / "sequences.json"
    COLLECTION_NAME = "esg_sequences"
    MIN_WORDS = 2  # Số từ tối thiểu cho một sequence
    BATCH_SIZE = 5  # Số sequences mỗi batch khi upload
    
    # Flags
    SKIP_JSON = False  # True nếu không cần lưu JSON
    SKIP_VECTORSTORE = False  # True nếu chỉ muốn tách sequences
    
    # Chạy pipeline
    success = run_sequences_pipeline(
        input_dir=str(INPUT_DIR),
        output_json=str(OUTPUT_JSON),
        collection_name=COLLECTION_NAME,
        min_words=MIN_WORDS,
        batch_size=BATCH_SIZE,
        skip_json=SKIP_JSON,
        skip_vectorstore=SKIP_VECTORSTORE
    )
    
    if success:
        print("\n🎉 Tất cả các bước đã hoàn thành thành công!")
    else:
        print("\n❌ Pipeline gặp lỗi. Vui lòng kiểm tra log.")

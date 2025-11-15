#!/usr/bin/env python3
"""
retrieval_benchmark.py

Hệ thống benchmark đánh giá retrieval cho các pages tương đồng nhất.
Bao gồm:
- Tạo ngân hàng câu hỏi ngẫu nhiên từ 4080 đoạn văn bản
- Hàm hybrid_search kết hợp similarity + BM25 ranking
- Phương thức benchmark đánh giá độ chính xác của pipeline retrieval
"""

import os
import sys
import json
import random
import time
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
from collections import defaultdict
import numpy as np

# Add project root to path
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document

# Load environment variables
load_dotenv()

# Import các hàm cần thiết
from helpers.init_qdrant import qdrant_client, create_qdrant_vectorstore
from helpers.vectorstore_from_sequences import retrieve_similar_sequences

@dataclass
class QuestionItem:
    """Cấu trúc cho một câu hỏi trong question bank"""
    question_id: str
    question: str
    ground_truth_sequence_id: str
    ground_truth_content: str
    file_name: str
    category: str  # Loại câu hỏi: 'direct', 'paraphrase', 'summary', 'related'

@dataclass
class RetrievalResult:
    """Kết quả retrieval cho một câu hỏi"""
    question_id: str
    retrieved_sequences: List[Dict[str, Any]]
    is_correct: bool
    rank: Optional[int]  # Vị trí của ground truth (1-based, None nếu không tìm thấy)
    similarity_score: float
    retrieval_time: float

@dataclass
class BenchmarkResults:
    """Kết quả tổng hợp benchmark"""
    accuracy: Dict[str, float]  # hit@k rates
    mrr: float  # Mean Reciprocal Rank
    avg_retrieval_time: float
    category_performance: Dict[str, Dict[str, float]]
    detailed_results: List[RetrievalResult]

class QuestionBankGenerator:
    """Tạo ngân hàng câu hỏi ngẫu nhiên từ sequences"""

    def __init__(self, sequences_file: str):
        self.sequences_file = sequences_file
        self.sequences = self._load_sequences()

        # Templates cho các loại câu hỏi
        self.question_templates = {
            'direct': [
                "Nội dung về '{keyword}' trong văn bản là gì?",
                "Thông tin liên quan đến '{keyword}' được đề cập như thế nào?",
                "Đoạn văn bản nói về '{keyword}' có nội dung gì?",
            ],
            'paraphrase': [
                "Bạn có thể giải thích về '{keyword}' theo cách khác không?",
                "Ý nghĩa của '{keyword}' trong tài liệu được thể hiện ra sao?",
                "Nội dung liên quan đến '{keyword}' được mô tả như thế nào?",
            ],
            'summary': [
                "Tóm tắt nội dung về '{keyword}' trong văn bản.",
                "Những điểm chính về '{keyword}' được đề cập là gì?",
                "Tóm lược thông tin về '{keyword}' từ tài liệu.",
            ],
            'related': [
                "Những vấn đề liên quan đến '{keyword}' được đề cập không?",
                "Ngoài '{keyword}', còn có thông tin gì liên quan?",
                "Các khía cạnh khác của '{keyword}' được nói đến như thế nào?",
            ]
        }

    def _load_sequences(self) -> List[Dict[str, Any]]:
        """Load sequences từ file JSON"""
        with open(self.sequences_file, 'r', encoding='utf-8') as f:
            return json.load(f)

    def _extract_keywords(self, content: str) -> List[str]:
        """Extract keywords từ content"""
        # Simple keyword extraction - có thể cải thiện với NLP
        words = content.split()
        keywords = []

        # Lấy các từ/cụm từ quan trọng
        for i in range(len(words)):
            if len(words[i]) > 3:  # Từ dài hơn 3 ký tự
                keywords.append(words[i])
            # Lấy bigram
            if i < len(words) - 1:
                bigram = f"{words[i]} {words[i+1]}"
                if len(bigram) > 6:
                    keywords.append(bigram)

        return list(set(keywords[:10]))  # Lấy tối đa 10 keywords

    def generate_question_bank(self, num_questions: int = 1000,
                             category_distribution: Optional[Dict[str, float]] = None,
                             min_words: int = 20) -> List[QuestionItem]:
        """
        Tạo question bank với phân phối categories

        Args:
            num_questions: Số lượng câu hỏi cần tạo
            category_distribution: Phân phối các loại câu hỏi
            min_words: Số từ tối thiểu trong sequence để được chọn
        """
        if category_distribution is None:
            category_distribution = {
                'direct': 0.4,
                'paraphrase': 0.3,
                'summary': 0.2,
                'related': 0.1
            }

        # Lọc sequences có đủ số từ
        filtered_sequences = [
            seq for seq in self.sequences
            if len(seq['content'].split()) >= min_words
        ]

        print(f"📊 Lọc sequences: {len(self.sequences)} → {len(filtered_sequences)} (min {min_words} words)")

        if len(filtered_sequences) == 0:
            print("❌ Không có sequence nào đủ điều kiện!")
            return []

        # Tính số lượng câu hỏi cho mỗi category
        category_counts = {}
        for cat, ratio in category_distribution.items():
            category_counts[cat] = int(num_questions * ratio)

        # Điều chỉnh để tổng bằng num_questions
        total = sum(category_counts.values())
        if total < num_questions:
            category_counts['direct'] += num_questions - total

        question_bank = []

        for category, count in category_counts.items():
            # Chọn ngẫu nhiên sequences cho category này từ filtered list
            available_sequences = min(count, len(filtered_sequences))
            selected_sequences = random.sample(filtered_sequences, available_sequences)

            for seq in selected_sequences:
                keywords = self._extract_keywords(seq['content'])
                if not keywords:
                    continue

                keyword = random.choice(keywords)
                template = random.choice(self.question_templates[category])

                question = template.format(keyword=keyword)

                question_item = QuestionItem(
                    question_id=f"{category}_{seq['sequence_id']}_{random.randint(1000, 9999)}",
                    question=question,
                    ground_truth_sequence_id=seq['sequence_id'],
                    ground_truth_content=seq['content'],
                    file_name=seq['file_name'],
                    category=category
                )

                question_bank.append(question_item)

        random.shuffle(question_bank)
        return question_bank

class HybridRetriever:
    """Hybrid search combining BM25 + Vector similarity"""

    def __init__(self, vectorstore, sequences_data: List[Dict[str, Any]]):
        self.vectorstore = vectorstore
        self.sequences_data = sequences_data

        # Tạo BM25 retriever
        docs = [
            Document(
                page_content=seq['content'],
                metadata={
                    'sequence_id': seq['sequence_id'],
                    'file_name': seq['file_name'],
                    'sequence_number': seq['sequence_number']
                }
            )
            for seq in sequences_data
        ]

        self.bm25_retriever = BM25Retriever.from_documents(docs)

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Hybrid search với BM25 + Vector similarity

        Args:
            query: Câu hỏi
            top_k: Số lượng kết quả trả về

        Returns:
            List các kết quả với metadata đầy đủ
        """
        # Thực hiện BM25 search
        bm25_results = self.bm25_retriever.get_relevant_documents(query, k=top_k)

        # Thực hiện vector search
        vector_results = self.vectorstore.similarity_search_with_score(query, k=top_k)

        # Kết hợp kết quả từ cả hai
        combined_results = []

        # Xử lý BM25 results
        for doc in bm25_results:
            sequence_id = doc.metadata.get('sequence_id')
            if sequence_id:
                # Tìm sequence gốc
                original_seq = None
                for seq in self.sequences_data:
                    if seq['sequence_id'] == sequence_id:
                        original_seq = seq
                        break

                if original_seq:
                    combined_results.append({
                        'doc': doc,
                        'source': 'bm25',
                        'sequence_id': sequence_id,
                        'original_seq': original_seq,
                        'score': 0.5  # Default score for BM25
                    })

        # Xử lý vector results - sử dụng Qdrant client để lấy full payload
        for doc, score in vector_results:
            # Lấy point ID từ metadata
            point_id = doc.metadata.get('_id')
            
            if point_id:
                try:
                    # Sử dụng Qdrant client để lấy full payload
                    from store.init_qdrant import qdrant_client
                    point_data = qdrant_client.retrieve(
                        collection_name='esg_sequences',
                        ids=[point_id],
                        with_payload=True,
                        with_vectors=False
                    )
                    
                    if point_data and len(point_data) > 0:
                        payload = point_data[0].payload
                        if payload:
                            sequence_id = payload.get('sequence_id') or payload.get('id')
                        else:
                            sequence_id = None
                        
                        if sequence_id:
                            # Tìm sequence gốc từ sequences_data
                            original_seq = None
                            for seq in self.sequences_data:
                                if seq['sequence_id'] == sequence_id:
                                    original_seq = seq
                                    break
                            
                            if original_seq:
                                combined_results.append({
                                    'doc': doc,
                                    'source': 'vector',
                                    'sequence_id': sequence_id,
                                    'original_seq': original_seq,
                                    'score': float(score)
                                })
                            else:
                                print(f"WARNING: Could not find original sequence for sequence_id: {sequence_id}")
                        else:
                            print(f"WARNING: No sequence_id in payload: {payload}")
                    else:
                        print(f"WARNING: Could not retrieve point data for ID: {point_id}")
                        
                except Exception as e:
                    print(f"WARNING: Error retrieving payload for point {point_id}: {e}")
            else:
                print(f"WARNING: No _id found in vector result metadata: {doc.metadata}")

        # Loại bỏ duplicates và lấy top kết quả
        seen_ids = set()
        unique_results = []

        # Sắp xếp theo score (vector results có score thực, BM25 có score mặc định)
        combined_results.sort(key=lambda x: x.get('score', 0.5), reverse=True)

        for result in combined_results:
            seq_id = result['sequence_id']
            if seq_id not in seen_ids:
                seen_ids.add(seq_id)
                unique_results.append(result)
                if len(unique_results) >= top_k:
                    break

        # Format kết quả cuối cùng
        formatted_results = []
        for result in unique_results:
            original_seq = result['original_seq']
            formatted_result = {
                'id': original_seq['sequence_id'],
                'content': original_seq['content'],
                'file_name': original_seq['file_name'],
                'sequence_number': original_seq['sequence_number'],
                'total_sequences_in_file': original_seq['total_sequences_in_file'],
                'similarity_score': result.get('score', 0.5),
                'source': result['source']
            }
            formatted_results.append(formatted_result)

        return formatted_results

class RetrievalBenchmark:
    """Benchmark đánh giá retrieval performance"""

    def __init__(self, retriever: HybridRetriever, question_bank: List[QuestionItem]):
        self.retriever = retriever
        self.question_bank = question_bank

    def evaluate_single_question(self, question_item: QuestionItem, top_k: int = 5) -> RetrievalResult:
        """Đánh giá một câu hỏi"""
        start_time = time.time()

        # Thực hiện retrieval
        retrieved_sequences = self.retriever.search(question_item.question, top_k=top_k)

        retrieval_time = time.time() - start_time

        # Kiểm tra xem ground truth có trong kết quả không
        rank = None
        is_correct = False
        similarity_score = 0.0

        for i, result in enumerate(retrieved_sequences, 1):
            if result['id'] == question_item.ground_truth_sequence_id:
                rank = i
                is_correct = True
                similarity_score = result['similarity_score']
                break

        return RetrievalResult(
            question_id=question_item.question_id,
            retrieved_sequences=retrieved_sequences,
            is_correct=is_correct,
            rank=rank,
            similarity_score=similarity_score,
            retrieval_time=retrieval_time
        )

    def run_benchmark(self, top_k_values: List[int] = [1, 3, 5, 10]) -> BenchmarkResults:
        """Chạy benchmark với tất cả câu hỏi"""
        print(f"🚀 Bắt đầu benchmark với {len(self.question_bank)} câu hỏi...")

        detailed_results = []
        retrieval_times = []

        for i, question_item in enumerate(self.question_bank, 1):
            if i % 50 == 0:
                print(f"📊 Đã xử lý {i}/{len(self.question_bank)} câu hỏi...")

            result = self.evaluate_single_question(question_item, max(top_k_values))
            detailed_results.append(result)
            retrieval_times.append(result.retrieval_time)

        # Tính toán metrics
        accuracy = {}
        for k in top_k_values:
            correct_count = sum(1 for r in detailed_results if r.rank and r.rank <= k)
            accuracy[f'hit@{k}'] = correct_count / len(detailed_results)

        # Mean Reciprocal Rank
        mrr = np.mean([
            1.0 / result.rank if result.rank else 0.0
            for result in detailed_results
        ])

        # Category performance
        category_performance = defaultdict(lambda: defaultdict(float))
        category_counts = defaultdict(int)

        for result in detailed_results:
            question_item = next(q for q in self.question_bank if q.question_id == result.question_id)
            category = question_item.category
            category_counts[category] += 1

            for k in top_k_values:
                if result.rank and result.rank <= k:
                    category_performance[category][f'hit@{k}'] += 1

        # Normalize category performance
        for category in category_performance:
            for k in top_k_values:
                if category_counts[category] > 0:
                    category_performance[category][f'hit@{k}'] /= category_counts[category]

        return BenchmarkResults(
            accuracy=dict(accuracy),
            mrr=float(mrr),
            avg_retrieval_time=float(np.mean(retrieval_times)),
            category_performance=dict(category_performance),
            detailed_results=detailed_results
        )

def print_benchmark_results(results: BenchmarkResults):
    """In kết quả benchmark"""
    print("\n" + "="*80)
    print("📊 KẾT QUẢ BENCHMARK RETRIEVAL")
    print("="*80)

    print("\n🎯 ACCURACY METRICS:")
    for metric, value in results.accuracy.items():
        print(f"   {metric}: {value:.1%}")

    print(f"\n📊 MRR: {results.mrr:.4f}")
    print(f"⏱️  Avg Retrieval Time: {results.avg_retrieval_time:.2f}s")

    print("\n📈 CATEGORY PERFORMANCE:")
    for category, metrics in results.category_performance.items():
        print(f"\n🏷️  {category.upper()}:")
        for metric, value in metrics.items():
            print(f"   {metric}: {value:.1%}")

    print("\n📋 THỐNG KÊ CHI TIẾT:")
    total_questions = len(results.detailed_results)
    correct_questions = sum(1 for r in results.detailed_results if r.is_correct)
    avg_similarity = np.mean([r.similarity_score for r in results.detailed_results if r.is_correct])

    print(f"   Tổng số câu hỏi: {total_questions}")
    print(f"   Số câu trả lời đúng: {correct_questions}")
    print(f"   Tỷ lệ chính xác: {correct_questions/total_questions:.1%}")
    print(f"   Điểm tương đồng trung bình: {avg_similarity:.4f}")

def save_benchmark_results(results: BenchmarkResults, output_file: str):
    """Lưu kết quả benchmark vào file JSON"""
    # Convert dataclasses to dicts
    results_dict = {
        'accuracy': results.accuracy,
        'mrr': results.mrr,
        'avg_retrieval_time': results.avg_retrieval_time,
        'category_performance': results.category_performance,
        'detailed_results': [
            {
                'question_id': r.question_id,
                'is_correct': r.is_correct,
                'rank': r.rank,
                'similarity_score': r.similarity_score,
                'retrieval_time': r.retrieval_time,
                'num_retrieved': len(r.retrieved_sequences)
            }
            for r in results.detailed_results
        ]
    }

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results_dict, f, ensure_ascii=False, indent=2)

    print(f"💾 Đã lưu kết quả benchmark vào {output_file}")

def main():
    """Main function để chạy benchmark"""
    print("🚀 KHỞI ĐỘNG HỆ THỐNG BENCHMARK RETRIEVAL")
    print("="*60)

    # Đường dẫn files
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent.parent  # Go up two levels to get to project root
    sequences_file = project_root / "src" / "data" / "push" / "merged_sequences.json"

    # Kiểm tra file sequences
    if not sequences_file.exists():
        print(f"❌ Không tìm thấy file sequences: {sequences_file}")
        return

    print(f"📂 File sequences: {sequences_file}")

    # 1. Tạo question bank
    print("\n📝 TẠO QUESTION BANK...")
    question_generator = QuestionBankGenerator(str(sequences_file))
    question_bank = question_generator.generate_question_bank(num_questions=500, min_words=20)

    print(f"✅ Đã tạo {len(question_bank)} câu hỏi")
    print("📊 Phân phối categories:")
    category_counts = defaultdict(int)
    for q in question_bank:
        category_counts[q.category] += 1

    for cat, count in category_counts.items():
        print(f"   {cat}: {count} câu hỏi")

    # 2. Khởi tạo retriever
    print("\n🔧 KHỞI TẠO HYBRID RETRIEVER...")
    try:
        vectorstore = create_qdrant_vectorstore('esg_sequences')
        print("✅ Đã kết nối đến vector store")
    except Exception as e:
        print(f"❌ Lỗi kết nối vector store: {e}")
        return

    # Load sequences data
    with open(sequences_file, 'r', encoding='utf-8') as f:
        sequences_data = json.load(f)

    retriever = HybridRetriever(vectorstore, sequences_data)
    print("✅ Đã khởi tạo hybrid retriever")

    # 3. Chạy benchmark
    print("\n🏃 CHẠY BENCHMARK...")
    benchmark = RetrievalBenchmark(retriever, question_bank)
    results = benchmark.run_benchmark(top_k_values=[1, 3, 5, 10])

    # 4. Hiển thị kết quả
    print_benchmark_results(results)

    # 5. Lưu kết quả
    output_file = project_root / "src" / "eval" / "benchmark_results.json"
    save_benchmark_results(results, str(output_file))

    print("\n✅ HOÀN THÀNH BENCHMARK!")
    print(f"📊 Kết quả được lưu tại: {output_file}")

if __name__ == "__main__":
    main()
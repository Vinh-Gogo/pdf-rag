#!/usr/bin/env python3
"""
Comprehensive Retrieval Benchmark System
Implements advanced evaluation metrics including IRT analysis and category-based evaluation
"""

import os
import json
import random
import time
import requests
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from collections import defaultdict
import numpy as np
import pandas as pd
from rank_bm25 import BM25Okapi
from qdrant_client import QdrantClient
from dotenv import load_dotenv
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

# Load environment variables
load_dotenv()

@dataclass
class QuestionItem:
    """Cấu trúc cho một câu hỏi trong question bank"""
    question_id: str
    question: str
    ground_truth_sequence_id: str
    ground_truth_content: str
    file_name: str
    category: str
    difficulty_level: str = "medium"  # easy, medium, hard
    keywords: List[str] = field(default_factory=list)

@dataclass
class RetrievalResult:
    """Kết quả retrieval cho một câu hỏi"""
    question_id: str
    question_category: str
    question_difficulty: str
    retrieved_sequences: List[Dict[str, Any]]
    is_correct: bool
    rank: Optional[int]
    similarity_score: float
    retrieval_time: float
    search_method: str = "hybrid"  # vector, bm25, hybrid

@dataclass
class BenchmarkResults:
    """Kết quả tổng hợp benchmark"""
    accuracy: Dict[str, float]
    mrr: float
    avg_retrieval_time: float
    category_performance: Dict[str, Dict[str, float]]
    difficulty_performance: Dict[str, Dict[str, float]]
    method_comparison: Dict[str, Dict[str, float]]
    irt_analysis: Dict[str, Any]
    detailed_results: List[RetrievalResult]
    statistical_summary: Dict[str, Any]

class EmbeddingClient:
    """Client để call embedding API"""

    def __init__(self):
        self.base_url = os.getenv('OPENAI_BASE_URL', 'http://localhost:8080')
        self.api_key = os.getenv('OPENAI_API_KEY_EMBED', 'dummy')

    def get_embedding(self, text: str) -> List[float]:
        """Get embedding cho một text"""
        try:
            response = requests.post(
                f"{self.base_url}/v1/embeddings",
                headers={
                    'Authorization': f'Bearer {self.api_key}',
                    'Content-Type': 'application/json'
                },
                json={
                    'input': text,
                    'model': 'Qwen/Qwen3-Embedding-0.6B'
                }
            )
            response.raise_for_status()
            data = response.json()
            return data['data'][0]['embedding']
        except Exception as e:
            print(f"❌ Error getting embedding: {e}")
            return []

class ComprehensiveRetriever:
    """Advanced retriever với multiple search methods"""

    def __init__(self, sequences_data: List[Dict[str, Any]]):
        self.sequences_data = sequences_data
        self.embedding_client = EmbeddingClient()

        # Khởi tạo Qdrant client
        qdrant_url = os.getenv('QDRANT_URL')
        qdrant_api_key = os.getenv('QDRANT_API_KEY')

        if not qdrant_url or not qdrant_api_key:
            raise ValueError("Missing QDRANT_URL or QDRANT_API_KEY")

        self.qdrant_client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)

        # Khởi tạo BM25
        corpus = [seq['content'] for seq in sequences_data]
        tokenized_corpus = [doc.split() for doc in corpus]
        self.bm25 = BM25Okapi(tokenized_corpus)

        # Map sequence_id to index
        self.id_to_index = {seq['sequence_id']: i for i, seq in enumerate(sequences_data)}

    def vector_search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Vector similarity search"""
        query_embedding = self.embedding_client.get_embedding(query)
        if not query_embedding:
            return []

        # Search trong Qdrant
        search_result = self.qdrant_client.search(
            collection_name='esg_sequences',
            query_vector=query_embedding,
            limit=top_k,
            with_payload=True,
            with_vectors=False
        )

        results = []
        for hit in search_result:
            payload = hit.payload
            results.append({
                'id': payload.get('id', ''),
                'content': payload.get('content', ''),
                'file_name': payload.get('file_name', ''),
                'sequence_number': payload.get('sequence_number', 0),
                'similarity_score': hit.score
            })

        return results

    def bm25_search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """BM25 keyword search với improved tokenization"""
        # Better tokenization - split by spaces and remove common words
        query_tokens = [word.lower().strip('.,!?()[]{}"\'')
                       for word in query.split()]
        query_tokens = [word for word in query_tokens
                       if len(word) > 2 and word not in [
                           'nội', 'dung', 'văn', 'bản', 'là', 'gì', 'thông', 'tin',
                           'liên', 'quan', 'đến', 'trong', 'theo', 'của', 'để',
                           'cho', 'với', 'hoặc', 'được', 'có', 'và', 'như', 'này',
                           'đó', 'làm', 'bằng', 'từ', 'về', 'qua', 'tại', 'năm'
                       ]]

        if not query_tokens:
            return []

        bm25_scores = self.bm25.get_scores(query_tokens)

        # Get top results
        top_indices = np.argsort(bm25_scores)[::-1][:top_k]

        results = []
        for idx in top_indices:
            seq = self.sequences_data[idx]
            score = float(bm25_scores[idx])
            if score > 0:  # Only include results with positive scores
                results.append({
                    'id': seq['sequence_id'],
                    'content': seq['content'],
                    'file_name': seq['file_name'],
                    'sequence_number': seq['sequence_number'],
                    'similarity_score': score
                })

        return results[:top_k]

    def hybrid_search(self, query: str, top_k: int = 5,
                     vector_weight: float = 0.2, bm25_weight: float = 0.8) -> List[Dict[str, Any]]:
        """Hybrid search với optimized parameters"""
        vector_results = self.vector_search(query, top_k=top_k*2)  # Get more candidates
        bm25_results = self.bm25_search(query, top_k=top_k*2)  # Get more candidates

        # Min-max normalize BM25 scores (best performing method)
        if bm25_results:
            bm25_scores = [r['similarity_score'] for r in bm25_results]
            bm25_min, bm25_max = min(bm25_scores), max(bm25_scores)
            if bm25_max > bm25_min:
                for result in bm25_results:
                    result['normalized_score'] = (result['similarity_score'] - bm25_min) / (bm25_max - bm25_min)
            else:
                for result in bm25_results:
                    result['normalized_score'] = 1.0 if result['similarity_score'] > 0 else 0.0

        # Combine scores with optimized weights
        combined_scores = {}

        # Add vector scores (already 0-1 normalized by cosine similarity)
        for result in vector_results:
            seq_id = result['id']
            combined_scores[seq_id] = {
                'score': result['similarity_score'] * vector_weight,
                'data': result
            }

        # Add normalized BM25 scores with higher weight
        for result in bm25_results:
            seq_id = result['id']
            normalized_bm25_score = result.get('normalized_score', 0.0)
            if seq_id in combined_scores:
                combined_scores[seq_id]['score'] += normalized_bm25_score * bm25_weight
            else:
                combined_scores[seq_id] = {
                    'score': normalized_bm25_score * bm25_weight,
                    'data': result
                }

        # Sort by combined score
        sorted_results = sorted(combined_scores.items(), key=lambda x: x[1]['score'], reverse=True)

        # Return top_k
        final_results = []
        for seq_id, data in sorted_results[:top_k]:
            result = data['data'].copy()
            result['similarity_score'] = data['score']
            final_results.append(result)

        return final_results

    def search(self, query: str, method: str = "hybrid", top_k: int = 5) -> List[Dict[str, Any]]:
        """Unified search interface"""
        if method == "vector":
            return self.vector_search(query, top_k)
        elif method == "bm25":
            return self.bm25_search(query, top_k)
        elif method == "hybrid":
            return self.hybrid_search(query, top_k)
        else:
            raise ValueError(f"Unknown search method: {method}")

class AdvancedQuestionGenerator:
    """Advanced question generator với difficulty levels và categories"""

    def __init__(self, sequences_file: str):
        self.sequences_file = sequences_file
        self.sequences = self._load_sequences()

    def _load_sequences(self) -> List[Dict[str, Any]]:
        """Load sequences từ file JSON"""
        with open(self.sequences_file, 'r', encoding='utf-8') as f:
            return json.load(f)

    def _extract_keywords(self, content: str) -> List[str]:
        """Extract meaningful keywords from content"""
        words = content.split()

        # Filter for meaningful words (Vietnamese)
        meaningful_words = []
        for word in words:
            word = word.strip('.,!?()[]{}"\'')
            if (len(word) >= 4 and  # At least 4 characters
                not word.isdigit() and  # Not just numbers
                not any(char.isdigit() for char in word) and  # No digits
                word not in ['trong', 'theo', 'của', 'là', 'để', 'từ', 'cho', 'với', 'hoặc', 'được', 'có', 'và', 'như', 'này']):  # Common stopwords
                meaningful_words.append(word)

        # Get unique keywords, prefer longer ones
        unique_words = list(set(meaningful_words))
        unique_words.sort(key=len, reverse=True)  # Longer words first

        return unique_words[:5]  # Return top 5 longest meaningful words

    def _classify_difficulty(self, content: str) -> str:
        """Classify question difficulty based on content characteristics"""
        length = len(content.split())
        keywords = self._extract_keywords(content)

        if length < 20 or len(keywords) < 2:
            return "easy"
        elif length < 50 or len(keywords) < 4:
            return "medium"
        else:
            return "hard"

    def generate_question_bank(self, num_questions: int = 100) -> List[QuestionItem]:
        """Tạo question bank với đa dạng categories và difficulty levels"""
        question_bank = []

        # Chọn ngẫu nhiên sequences
        selected_sequences = random.sample(self.sequences, min(num_questions, len(self.sequences)))

        for i, seq in enumerate(selected_sequences):
            content = seq['content']
            keywords = self._extract_keywords(content)
            difficulty = self._classify_difficulty(content)

            # Tạo câu hỏi đa dạng hơn
            question_types = [
                ("context", self._create_context_question),
                ("definition", self._create_definition_question),
                ("relationship", self._create_relationship_question),
                ("factual", self._create_factual_question)
            ]

            # Weight question types by difficulty
            if difficulty == "easy":
                weights = [0.4, 0.3, 0.2, 0.1]  # More context questions
            elif difficulty == "medium":
                weights = [0.3, 0.2, 0.3, 0.2]  # Balanced
            else:  # hard
                weights = [0.2, 0.1, 0.4, 0.3]  # More relationship and factual

            question_type_name, question_func = random.choices(question_types, weights=weights)[0]
            question = question_func(content, keywords)

            if not question:
                # Fallback to simple keyword question
                if keywords:
                    keyword = random.choice(keywords)
                    question = f"Nội dung về '{keyword}' trong văn bản là gì?"
                    question_type_name = "keyword"

            if question:
                question_item = QuestionItem(
                    question_id=f"q_{i+1}",
                    question=question,
                    ground_truth_sequence_id=seq['sequence_id'],
                    ground_truth_content=content,
                    file_name=seq['file_name'],
                    category=question_type_name,
                    difficulty_level=difficulty,
                    keywords=keywords
                )
                question_bank.append(question_item)

        return question_bank

    def _create_context_question(self, content: str, keywords: List[str]) -> Optional[str]:
        """Tạo câu hỏi về context"""
        if not keywords:
            return None

        keyword = random.choice(keywords[:3])  # Use top keywords
        templates = [
            f"Trong văn bản, '{keyword}' được đề cập trong bối cảnh nào?",
            f"Văn bản nói về '{keyword}' như thế nào?",
            f"Ý nghĩa của '{keyword}' trong đoạn văn này là gì?"
        ]
        return random.choice(templates)

    def _create_definition_question(self, content: str, keywords: List[str]) -> Optional[str]:
        """Tạo câu hỏi về định nghĩa"""
        if "là" in content.lower() or "được định nghĩa" in content.lower():
            if keywords:
                keyword = random.choice(keywords[:2])
                return f"'{keyword}' được định nghĩa như thế nào trong văn bản?"

        return None

    def _create_relationship_question(self, content: str, keywords: List[str]) -> Optional[str]:
        """Tạo câu hỏi về mối quan hệ"""
        if len(keywords) >= 2:
            kw1, kw2 = random.sample(keywords[:4], 2)
            templates = [
                f"Mối quan hệ giữa '{kw1}' và '{kw2}' trong văn bản là gì?",
                f"Văn bản liên kết '{kw1}' với '{kw2}' như thế nào?"
            ]
            return random.choice(templates)

        return None

    def _create_factual_question(self, content: str, keywords: List[str]) -> Optional[str]:
        """Tạo câu hỏi factual"""
        if keywords:
            keyword = random.choice(keywords)
            templates = [
                f"Thông tin về '{keyword}' trong văn bản là gì?",
                f"Văn bản đề cập đến '{keyword}' với nội dung nào?"
            ]
            return random.choice(templates)

        return None

class IRTAnalyzer:
    """Item Response Theory analysis for question difficulty and discrimination"""

    def __init__(self):
        pass

    def analyze_irt(self, results: List[RetrievalResult]) -> Dict[str, Any]:
        """Perform IRT analysis on retrieval results"""
        # Group results by question
        question_stats = defaultdict(list)

        for result in results:
            question_stats[result.question_id].append({
                'correct': result.is_correct,
                'rank': result.rank,
                'difficulty': result.question_difficulty,
                'category': result.question_category
            })

        # Calculate item parameters
        item_parameters = {}
        for question_id, responses in question_stats.items():
            correct_responses = sum(1 for r in responses if r['correct'])
            total_responses = len(responses)
            difficulty = correct_responses / total_responses

            # Calculate discrimination (point-biserial correlation)
            correct_scores = [r['rank'] or 100 for r in responses if r['correct']]  # High rank = low score
            incorrect_scores = [r['rank'] or 100 for r in responses if not r['correct']]

            if correct_scores and incorrect_scores:
                all_scores = correct_scores + incorrect_scores
                correct_mean = np.mean(correct_scores)
                incorrect_mean = np.mean(incorrect_scores)
                total_std = np.std(all_scores)

                if total_std > 0:
                    discrimination = (correct_mean - incorrect_mean) / total_std
                else:
                    discrimination = 0
            else:
                discrimination = 0

            item_parameters[question_id] = {
                'difficulty': difficulty,
                'discrimination': discrimination,
                'total_responses': total_responses,
                'correct_responses': correct_responses
            }

        # Calculate ability estimates (simplified)
        ability_estimates = {}
        for result in results:
            # Simple ability estimate based on performance
            ability_estimates[result.question_id] = item_parameters[result.question_id]['difficulty']

        return {
            'item_parameters': item_parameters,
            'ability_estimates': ability_estimates,
            'summary': {
                'avg_difficulty': np.mean([p['difficulty'] for p in item_parameters.values()]),
                'avg_discrimination': np.mean([p['discrimination'] for p in item_parameters.values()]),
                'difficulty_distribution': {
                    'easy': len([p for p in item_parameters.values() if p['difficulty'] > 0.7]),
                    'medium': len([p for p in item_parameters.values() if 0.4 <= p['difficulty'] <= 0.7]),
                    'hard': len([p for p in item_parameters.values() if p['difficulty'] < 0.4])
                }
            }
        }

class ComprehensiveBenchmark:
    """Comprehensive benchmark với advanced analytics"""

    def __init__(self, retriever: ComprehensiveRetriever, question_bank: List[QuestionItem]):
        self.retriever = retriever
        self.question_bank = question_bank
        self.irt_analyzer = IRTAnalyzer()

    def evaluate_single_question(self, question_item: QuestionItem,
                               search_method: str = "hybrid", top_k: int = 5) -> RetrievalResult:
        """Đánh giá một câu hỏi với specified search method"""
        start_time = time.time()

        # Thực hiện search
        retrieved_sequences = self.retriever.search(question_item.question,
                                                   method=search_method, top_k=top_k)

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
            question_category=question_item.category,
            question_difficulty=question_item.difficulty_level,
            retrieved_sequences=retrieved_sequences,
            is_correct=is_correct,
            rank=rank,
            similarity_score=similarity_score,
            retrieval_time=retrieval_time,
            search_method=search_method
        )

    def run_comprehensive_benchmark(self, top_k_values: List[int] = [1, 3, 5],
                                  search_methods: List[str] = ["vector", "bm25", "hybrid"]) -> BenchmarkResults:
        """Chạy comprehensive benchmark với multiple methods"""
        print(f"Chạy comprehensive benchmark với {len(self.question_bank)} câu hỏi...")

        all_results = []

        # Test từng search method
        for method in search_methods:
            print(f"\n--- Testing {method.upper()} search ---")
            method_results = []

            for i, question_item in enumerate(self.question_bank, 1):
                if i % 20 == 0:
                    print(f"  Đã xử lý {i}/{len(self.question_bank)} câu hỏi...")

                result = self.evaluate_single_question(question_item, search_method=method, top_k=max(top_k_values))
                result.search_method = method
                method_results.append(result)
                all_results.append(result)

            # Quick stats for this method
            correct_count = sum(1 for r in method_results if r.is_correct)
            print(f"  {method.upper()}: {correct_count}/{len(method_results)} correct ({correct_count/len(method_results):.1%})")

        # Tính toán metrics tổng hợp
        accuracy = {}
        for k in top_k_values:
            correct_count = sum(1 for r in all_results if r.rank and r.rank <= k)
            accuracy[f'hit@{k}'] = correct_count / len(all_results)

        # Mean Reciprocal Rank
        mrr = np.mean([
            1.0 / result.rank if result.rank else 0.0
            for result in all_results
        ])

        # Category performance
        category_performance = defaultdict(lambda: defaultdict(float))
        category_counts = defaultdict(int)

        for result in all_results:
            category_counts[result.question_category] += 1
            for k in top_k_values:
                if result.rank and result.rank <= k:
                    category_performance[result.question_category][f'hit@{k}'] += 1

        # Convert to percentages
        for category in category_performance:
            for metric in category_performance[category]:
                category_performance[category][metric] /= category_counts[category]

        # Difficulty performance
        difficulty_performance = defaultdict(lambda: defaultdict(float))
        difficulty_counts = defaultdict(int)

        for result in all_results:
            difficulty_counts[result.question_difficulty] += 1
            for k in top_k_values:
                if result.rank and result.rank <= k:
                    difficulty_performance[result.question_difficulty][f'hit@{k}'] += 1

        # Convert to percentages
        for difficulty in difficulty_performance:
            for metric in difficulty_performance[difficulty]:
                difficulty_performance[difficulty][metric] /= difficulty_counts[difficulty]

        # Method comparison
        method_comparison = defaultdict(lambda: defaultdict(float))
        method_counts = defaultdict(int)

        for result in all_results:
            method_counts[result.search_method] += 1
            for k in top_k_values:
                if result.rank and result.rank <= k:
                    method_comparison[result.search_method][f'hit@{k}'] += 1

        # Convert to percentages
        for method in method_comparison:
            for metric in method_comparison[method]:
                method_comparison[method][metric] /= method_counts[method]

        # IRT Analysis
        irt_analysis = self.irt_analyzer.analyze_irt(all_results)

        # Statistical summary
        retrieval_times = [r.retrieval_time for r in all_results]
        statistical_summary = {
            'retrieval_time_stats': {
                'mean': float(np.mean(retrieval_times)),
                'std': float(np.std(retrieval_times)),
                'min': float(np.min(retrieval_times)),
                'max': float(np.max(retrieval_times)),
                'median': float(np.median(retrieval_times))
            },
            'question_distribution': dict(category_counts),
            'difficulty_distribution': dict(difficulty_counts),
            'method_distribution': dict(method_counts),
            'total_questions': len(self.question_bank),
            'total_evaluations': len(all_results)
        }

        return BenchmarkResults(
            accuracy=dict(accuracy),
            mrr=float(mrr),
            avg_retrieval_time=float(np.mean(retrieval_times)),
            category_performance=dict(category_performance),
            difficulty_performance=dict(difficulty_performance),
            method_comparison=dict(method_comparison),
            irt_analysis=irt_analysis,
            detailed_results=all_results,
            statistical_summary=statistical_summary
        )

def print_comprehensive_results(results: BenchmarkResults):
    """In comprehensive results"""
    print("\n" + "="*100)
    print("COMPREHENSIVE RETRIEVAL BENCHMARK RESULTS")
    print("="*100)

    print("\n📊 OVERALL ACCURACY METRICS:")
    for metric, value in results.accuracy.items():
        print(".1%")

    print(".4f")
    print(".2f")

    print("\n📈 METHOD COMPARISON:")
    methods = ['vector', 'bm25', 'hybrid']
    print("<8")
    print("-" * 50)
    for method in methods:
        if method in results.method_comparison:
            hit1 = results.method_comparison[method].get('hit@1', 0)
            hit3 = results.method_comparison[method].get('hit@3', 0)
            hit5 = results.method_comparison[method].get('hit@5', 0)
            print("<8")

    print("\n🏷️  CATEGORY PERFORMANCE:")
    for category, metrics in results.category_performance.items():
        print(f"\n{category.upper()}:")
        for metric, value in metrics.items():
            print(".1%")

    print("\n⚡ DIFFICULTY PERFORMANCE:")
    for difficulty, metrics in results.difficulty_performance.items():
        print(f"\n{difficulty.upper()}:")
        for metric, value in metrics.items():
            print(".1%")

    print("\n🧠 IRT ANALYSIS SUMMARY:")
    irt = results.irt_analysis['summary']
    print(".3f")
    print(".3f")
    print(f"Difficulty Distribution: {irt['difficulty_distribution']}")

    print("\n📈 STATISTICAL SUMMARY:")
    stats = results.statistical_summary
    rt_stats = stats['retrieval_time_stats']
    print(f"Retrieval Time - Mean: {rt_stats['mean']:.3f}s, Std: {rt_stats['std']:.3f}s")
    print(f"Question Distribution: {stats['question_distribution']}")
    print(f"Total Evaluations: {stats['total_evaluations']}")

def main():
    """Main function"""
    print("COMPREHENSIVE RETRIEVAL BENCHMARK")
    print("="*50)

    # Đường dẫn files
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent
    sequences_file = project_root / "data" / "reports" / "merged_sequences.json"

    # Kiểm tra file sequences
    if not sequences_file.exists():
        print(f"❌ Không tìm thấy file sequences: {sequences_file}")
        return

    print(f"File sequences: {sequences_file}")

    # Load sequences data
    with open(sequences_file, 'r', encoding='utf-8') as f:
        sequences_data = json.load(f)

    print(f"Loaded {len(sequences_data)} sequences")

    # 1. Tạo advanced question bank
    print("\n🔍 TẠO ADVANCED QUESTION BANK...")
    question_generator = AdvancedQuestionGenerator(str(sequences_file))
    question_bank = question_generator.generate_question_bank(num_questions=50)  # Test với 50 câu

    print(f"✅ Đã tạo {len(question_bank)} câu hỏi")
    category_counts = defaultdict(int)
    difficulty_counts = defaultdict(int)
    for q in question_bank:
        category_counts[q.category] += 1
        difficulty_counts[q.difficulty_level] += 1
    print(f"   Categories: {dict(category_counts)}")
    print(f"   Difficulties: {dict(difficulty_counts)}")

    # 2. Khởi tạo retriever
    print("\n🚀 KHỞI TẠO COMPREHENSIVE RETRIEVER...")
    try:
        retriever = ComprehensiveRetriever(sequences_data)
        print("✅ Đã khởi tạo retriever")
    except Exception as e:
        print(f"❌ Lỗi khởi tạo retriever: {e}")
        return

    # 3. Chạy comprehensive benchmark
    print("\n⚡ CHẠY COMPREHENSIVE BENCHMARK...")
    benchmark = ComprehensiveBenchmark(retriever, question_bank)
    results = benchmark.run_comprehensive_benchmark(
        top_k_values=[3, 5, 10],
        search_methods=["vector", "bm25", "hybrid"]
    )

    # 4. Hiển thị kết quả
    print_comprehensive_results(results)

    print("\n🎉 HOÀN THÀNH COMPREHENSIVE BENCHMARK!")

if __name__ == "__main__":
    main()
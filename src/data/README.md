# Data Directory Structure

Thư mục này chứa tất cả kết quả xử lý dữ liệu từ Vietnamese text processing pipeline.

## 📁 Cấu trúc thư mục:

```
src/data/
├── DATA_INDEX.json              # Index file mô tả cấu trúc
├── contents/                    # Original content files
│   ├── page_1.txt
│   ├── page_2.txt
│   └── ...
└── results/                     # Kết quả xử lý
    ├── processing/              # Vietnamese preprocessing results
    │   ├── all_contents_processing_results.json
    │   └── contents_processing_summary.json
    ├── search/                  # BM25 search files
    │   ├── processed_sequences_bm25.json
    │   └── bm25_search_test_results.json
    ├── testing/                 # Testing results
    │   ├── test_results.json
    │   ├── detailed_test_results.json
    │   ├── pipeline_test_results.json
    │   └── preprocessing_variations_test.json
    └── reports/                 # Documentation & reports
        ├── TESTING_REPORT.md
        └── FINAL_COMPLETION_REPORT.md
```

## 🚀 Sử dụng:

### Cho BM25 Search:
```python
with open("src/data/results/search/processed_sequences_bm25.json", 'r', encoding='utf-8') as f:
    sequences_data = json.load(f)
```

### Cho Analysis:
```python
with open("src/data/results/processing/all_contents_processing_results.json", 'r', encoding='utf-8') as f:
    full_results = json.load(f)
```

### Xem Reports:
- `src/data/results/reports/TESTING_REPORT.md` - Comprehensive testing report
- `src/data/results/reports/FINAL_COMPLETION_REPORT.md` - Final completion report

## 📊 Thống kê:
- 168 pages được xử lý
- 1,658 sequences được tạo ra
- 100% success rate
- BM25 search ready

# Memory Bank - PDF-RAG Project

This directory contains the comprehensive Memory Bank for the PDF-RAG project. Since I (Cline) have memory that resets between sessions, this documentation is my single source of truth for understanding the project, continuing work, and making informed decisions.

## ÔøΩÔøΩ Reading Order

Start with these files in this order to understand the full context:

1. **projectbrief.md** - Start here
   - Foundation document with core requirements
   - Project objectives and scope
   - Success metrics

2. **productContext.md** - Then read this
   - Why the project exists
   - Problems it solves
   - How it should work
   - Current baseline results

3. **systemPatterns.md** - Architecture understanding
   - System overview and design patterns
   - Key components and data flow
   - Technical decisions and rationale

4. **techContext.md** - Technical details
   - Technology stack and dependencies
   - File formats and directory structure
   - Configuration parameters
   - Development commands

5. **activeContext.md** - Current session work
   - What's completed and in progress
   - Active decisions being made
   - Known issues and workarounds
   - Code style and preferences

6. **progress.md** - Project evolution
   - What works and what's left
   - How decisions evolved
   - Performance timeline
   - Next steps and recommendations

## Ì¥ë Key Information at a Glance

### Project Goal
Generate retrieval test questions from Vietnamese business documents (BIWASE bulletins) and evaluate semantic search accuracy.

### Current Status
- ‚úÖ 273 test questions generated (~8 per file)
- ‚úÖ File-level retrieval evaluation working
- ‚úÖ Baseline accuracy@10: 89.4%

### Core Files
- `generate_retrieval_tests.py` - Question generation (Qwen-1.7B)
- `measure_retrieval_accuracy.py` - Evaluation pipeline
- `_pdf_md/` - 34 markdown source files
- `tests/data/retrieval_test_cases.jsonl` - Generated questions
- `tests/data/evaluation_results.json` - Evaluation metrics

### Next Session Steps
1. Read ALL Memory Bank files (you are here!)
2. Verify current state matches documented status
3. Check evaluation metrics in evaluation_results.json
4. Plan improvements based on findings
5. Continue development work

## Ì≥ù Updating This Memory Bank

After completing work or when triggered by "update memory bank" request:

1. **Always review ALL files** - don't skip any
2. **Update activeContext.md** - current work and decisions
3. **Update progress.md** - what's done and what's next
4. **Document patterns** - code style, architecture decisions
5. **Include learnings** - what we discovered this session

## ÌæØ Quick Reference

| Question | Answer |
|----------|--------|
| Where's the main code? | generate_retrieval_tests.py, measure_retrieval_accuracy.py |
| What model is used? | Qwen-1.7B for questions, QwenEmbedding for retrieval |
| How many test questions? | 273 (~8 per file, target is 100) |
| Current accuracy? | 42.1% @1, 64.5% @3, 89.4% @10 |
| Python version? | 3.12.9 |
| GPU? | Auto-detected, bfloat16 precision |
| Language? | Vietnamese (BIWASE bulletins) |
| Evaluation scope? | File-level (34 documents), not chunk-level |

---

Created: December 10, 2025  
Last Updated: December 10, 2025  
Status: Active Project

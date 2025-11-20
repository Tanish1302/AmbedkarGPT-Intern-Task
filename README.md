# AmbedkarGPT - RAG Evaluation Framework
**Assignment 2 - Comprehensive Evaluation System**

## 📋 Overview

This project implements a comprehensive evaluation framework for the AmbedkarGPT RAG (Retrieval-Augmented Generation) system. The evaluation measures system performance across multiple documents using standard NLP metrics and comparative chunking analysis.

**Company:** Kalpit Pvt Ltd, UK  
**Assignment:** AI Intern Hiring - Phase 2  
**Candidate:** [Your Name]

## 🎯 Assignment Objectives

✅ Implement 8 comprehensive evaluation metrics  
✅ Compare 3 different chunking strategies  
✅ Analyze performance across 25 test questions  
✅ Identify optimal configuration and failure modes  
✅ Provide actionable recommendations

## 📊 Evaluation Metrics Implemented

### Retrieval Metrics
- **Hit Rate**: Whether any relevant document was retrieved
- **Mean Reciprocal Rank (MRR)**: Ranking quality of relevant documents
- **Precision@K**: Precision of top-K retrieved documents

### Answer Quality Metrics
- **Answer Relevance**: Semantic relevance of answer to question
- **Faithfulness**: Factual consistency with source documents
- **ROUGE-L Score**: Longest common subsequence with ground truth

### Semantic Metrics
- **Cosine Similarity**: Embedding-based semantic similarity
- **BLEU Score**: N-gram overlap with reference answer

## 📁 Project Structure

```
AmbedkarGPT-Intern-Task/
├── corpus/                      # Document corpus (6 speeches)
│   ├── speech1.txt             # Annihilation of Caste
│   ├── speech2.txt             # The Buddha and His Dhamma
│   ├── speech3.txt             # States and Minorities
│   ├── speech4.txt             # Waiting for a Visa
│   ├── speech5.txt             # Pakistan or Partition
│   └── speech6.txt             # The Untouchables
├── evaluation.py                # Main evaluation script
├── test_dataset.json           # 25 test Q&A pairs
├── test_results.json           # Evaluation results (generated)
├── results_analysis.md         # Detailed findings and recommendations
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

## 🚀 Setup Instructions

### Prerequisites

1. **Python 3.8+** installed
2. **Ollama** installed with Mistral 7B model
3. **Git** for version control

### Step 1: Install Ollama

```bash
# For Linux/Mac
curl -fsSL https://ollama.ai/install.sh | sh

# Pull Mistral 7B model
ollama pull mistral

# Verify installation
ollama list
```

### Step 2: Clone Repository

```bash
git clone https://github.com/[your-username]/AmbedkarGPT-Intern-Task.git
cd AmbedkarGPT-Intern-Task
```

### Step 3: Set Up Python Environment

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Linux/Mac:
source venv/bin/activate
# On Windows:
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Step 4: Download NLTK Data (First Time Only)

```python
# Open Python shell
python

# Download required data
import nltk
nltk.download('punkt')
exit()
```

## 🏃 Running the Evaluation

### Quick Start

```bash
# Ensure Ollama is running
ollama serve

# In another terminal, run evaluation
python evaluation.py
```

### What Happens During Evaluation

1. **Loads** 6 documents from corpus/
2. **Tests** 3 chunking strategies:
   - Small chunks: 200-300 characters
   - Medium chunks: 500-600 characters
   - Large chunks: 800-1000 characters
3. **Evaluates** all 25 test questions for each strategy
4. **Calculates** 8 metrics per question
5. **Generates** test_results.json with complete results
6. **Prints** summary comparison

### Expected Runtime

- **Total time:** 30-60 minutes (depending on hardware)
- **Per strategy:** 10-20 minutes
- **Per question:** ~30-45 seconds

## 📈 Understanding Results

### test_results.json Structure

```json
{
  "timestamp": "2024-...",
  "chunking_strategies": {
    "small_chunks": {
      "aggregate_metrics": {
        "avg_hit_rate": 0.85,
        "avg_mrr": 0.72,
        "avg_rouge_l": 0.45,
        ...
      },
      "question_results": [...]
    },
    "medium_chunks": {...},
    "large_chunks": {...}
  }
}
```

### Key Metrics to Compare

| Metric | Good Score | Interpretation |
|--------|------------|----------------|
| Hit Rate | > 0.8 | Retrieves relevant docs |
| MRR | > 0.7 | Ranks relevant docs high |
| Precision@K | > 0.5 | Relevant docs in top-K |
| ROUGE-L | > 0.4 | Answer matches ground truth |
| Cosine Similarity | > 0.6 | Semantically similar |
| Faithfulness | > 0.7 | Stays true to context |

## 🔍 Analyzing Results

### Generate Analysis Report

After running evaluation.py, analyze results:

```python
# Create analysis script or manually review test_results.json
import json

with open('test_results.json', 'r') as f:
    results = json.load(f)

# Compare strategies
for strategy, data in results['chunking_strategies'].items():
    print(f"\n{strategy}:")
    agg = data['aggregate_metrics']
    print(f"  Hit Rate: {agg['avg_hit_rate']:.3f}")
    print(f"  ROUGE-L: {agg['avg_rouge_l']:.3f}")
    print(f"  Cosine Sim: {agg['avg_cosine_similarity']:.3f}")
```

### Questions to Answer

1. **Which chunking strategy performs best?**
   - Compare avg_hit_rate, avg_rouge_l, avg_cosine_similarity

2. **What is current accuracy?**
   - Overall hit rate and precision@k

3. **Common failure modes?**
   - Questions with low scores
   - Unanswerable question handling
   - Comparative vs factual performance

4. **Improvement recommendations?**
   - Optimal chunk size
   - Retrieval parameters (k value)
   - Prompt engineering needs

## 🛠️ Troubleshooting

### Issue: Ollama Connection Error

```bash
# Start Ollama server
ollama serve

# Check if running
curl http://localhost:11434/api/version
```

### Issue: CUDA/GPU Errors

```bash
# Use CPU-only mode (automatic fallback)
# Or install CUDA-compatible PyTorch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Issue: Memory Errors

```python
# Reduce batch processing in evaluation.py
# Or process fewer questions at once
```

### Issue: ChromaDB Persistence Errors

```bash
# Clear existing databases
rm -rf chroma_db_*

# Rerun evaluation
python evaluation.py
```

## 📚 Technical Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| Framework | LangChain | RAG orchestration |
| Vector DB | ChromaDB | Document embeddings storage |
| Embeddings | sentence-transformers/all-MiniLM-L6-v2 | Text to vectors |
| LLM | Ollama + Mistral 7B | Answer generation |
| Metrics | rouge-score, nltk, scikit-learn | Evaluation |

## 🎓 Evaluation Methodology

### Chunking Strategies

1. **Small Chunks (200-300 chars)**
   - Precise context retrieval
   - More chunks, more granular
   - Risk: Context fragmentation

2. **Medium Chunks (500-600 chars)**
   - Balanced approach
   - Sufficient context
   - Moderate chunk count

3. **Large Chunks (800-1000 chars)**
   - Maximum context
   - Fewer, broader chunks
   - Risk: Noise in retrieval

### Test Dataset Composition

- **Total Questions:** 25
- **Factual:** 16 (64%)
- **Comparative:** 4 (16%)
- **Conceptual:** 2 (8%)
- **Unanswerable:** 3 (12%)

## 📝 Deliverables Checklist

- [x] evaluation.py - Complete evaluation script
- [x] test_dataset.json - 25 test Q&A pairs
- [x] corpus/ - 6 document files
- [x] requirements.txt - All dependencies
- [x] README.md - Complete documentation
- [ ] test_results.json - Generated after running
- [ ] results_analysis.md - Analysis and recommendations

## 🔗 Repository Information

**Repository Name:** AmbedkarGPT-Intern-Task  
**Repository URL:** [Your GitHub URL]  
**Contact:** [Your Email]

## 📧 Submission

**Hiring Manager:** Kalpit Pvt Ltd, UK  
**Email:** kalpiksingh2005@gmail.com  
**Deadline:** 4 days from assignment receipt

## 🙏 Acknowledgments

- Dr. B.R. Ambedkar for the inspirational texts
- LangChain community for the excellent framework
- Ollama team for local LLM capabilities

---

**Note:** This is an internship assignment project for Kalpit Pvt Ltd, UK. All code is original and demonstrates understanding of RAG evaluation principles.

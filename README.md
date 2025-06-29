# Retrieval Evaluation for RAG Systems

This project focuses specifically on evaluating the **Retrieval** component of RAG (Retrieval-Augmented Generation) systems. While RAG consists of both retrieval and generation components, this benchmark focuses solely on measuring the quality of document retrieval. This project implements semantic search evaluation using local vector storage with FAISS, as opposed to the in-memory approach used in the original BEIR benchmark. This implementation enables efficient handling of large-scale evaluations by storing embeddings on disk.


## What This Evaluates

### Retrieval Evaluation (✓ Covered)
- Document embedding quality
- Semantic search accuracy
- Retrieval metrics (NDCG@k, Precision@k, Recall@k)
- Ranking effectiveness
- Vector similarity search performance

### RAG Components Not Evaluated (❌ Not Covered)
- Text generation quality
- Answer synthesis
- Hallucination detection
- Response accuracy
- Context integration

## Why Focus on Retrieval?

1. **Foundation of RAG**
   - Retrieval quality directly impacts generation quality
   - Poor retrieval cannot be compensated by good generation
   - Efficient retrieval is crucial for system performance

2. **Quantitative Metrics**
   - Retrieval can be evaluated with standard IR metrics
   - NDCG@10 provides clear comparison between models
   - Results are reproducible and objective

3. **Separation of Concerns**
   - Allows focused optimization of retrieval models
   - Independent of LLM choice for generation
   - Clearer performance bottleneck identification

## Overview

The system evaluates semantic search models by:
1. Computing document embeddings using transformer models
2. Storing vectors locally using FAISS indices
3. Computing retrieval metrics (NDCG@k, Precision@k, Recall@k)
4. Aggregating results across multiple datasets

### Key Features
- Local vector storage using FAISS
- Support for both CPU and GPU computation
- Batch processing to handle memory constraints
- Comprehensive evaluation metrics
- Model-specific vector storage organization

## Installation

### Requirements
```bash
pip install sentence-transformers torch datasets pandas numpy tqdm
```

### FAISS Installation
Choose one based on your hardware:

For CPU-only:
```bash
pip install faiss-cpu
```

For GPU support:
```bash
pip install faiss-gpu
```

Note: `faiss-gpu` requires CUDA to be installed on your system.

## Usage

### Basic Usage
```bash
python benchmark.py --models_file models.txt --output results.csv --batch_size 32
```

### Command Line Arguments
- `--models_file`: Path to file containing model names (default: models.txt)
- `--batch_size`: Batch size for encoding (default: 32)
- `--output`: Path to output file (default: results.csv)
- `--force_recompute`: Force recomputation of vectors

### Models File Format
Create a `models.txt` file with model names, one per line:
```text
BAAI/bge-m3
intfloat/multilingual-e5-base
Snowflake/snowflake-arctic-embed-l-v2.0
```

## Implementation Details

### Vector Storage
Unlike the original BEIR implementation which keeps vectors in memory, this implementation:
1. Computes embeddings in batches
2. Stores vectors in FAISS indices on disk
3. Creates separate storage directories for each model
4. Loads vectors only when needed for evaluation

### Directory Structure
```
vector_store/
    model1_name/
        dataset1_vectors.faiss
        dataset1_corpus_ids.pkl
    model2_name/
        dataset2_vectors.faiss
        dataset2_corpus_ids.pkl
results/
    model1_results_timestamp.csv
    model2_results_timestamp.csv
final_results.csv
```

### NDCG@10 Calculation
The average NDCG@10 score is calculated by:
1. Computing NDCG@10 for each query in each dataset
2. Averaging NDCG@10 scores across all queries in a dataset
3. Computing the mean across all datasets for each model
4. Converting to percentage (multiplying by 100)

## Results

Example evaluation results for different models:

| Model | Average NDCG@10 |
|-------|----------------|
| BAAI/bge-m3 | 65.80% |
| Snowflake/snowflake-arctic-embed-l-v2.0 | 64.52% |
| openai/text-embedding-3-large | 62.35% |
| intfloat/multilingual-e5-large |  62.29% |
| intfloat/multilingual-e5-base | 61.38% |
| cohere/embed-multilingual-v3.0 | 60.54% |
| intfloat/multilingual-e5-small | 58.98% |
| sentence-transformers/LaBSE | 51.51% |
| LocalDoc/TEmA-small | 50.68% |
| LocalDoc/az-en-MiniLM-L6-v2 | 49.88% |
| sentence-transformers/paraphrase-multilingual-mpnet-base-v2 | 47.61% |
| sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 | 31.09% |
| sentence-transformers/all-MiniLM-L6-v2 | 24.25% |
## License

MIT License

## Authors

LocalDoc Team

---

For more information about the underlying technologies:
- [FAISS](https://github.com/facebookresearch/faiss)
- [BEIR Benchmark](https://github.com/beir-cellar/beir)

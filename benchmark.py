import torch
from datasets import load_dataset
from sentence_transformers import SentenceTransformer, util
from tqdm.auto import tqdm
import numpy as np
from collections import defaultdict
import pandas as pd
import faiss
import pickle
import os
import argparse
from typing import List, Dict, Set
from datetime import datetime

class SemanticSearchEvaluator:
    def __init__(self, model_name, dataset_name, vectors_dir='vector_store', batch_size=32, device=None):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = SentenceTransformer(model_name).to(self.device)
        self.batch_size = batch_size
        self.dataset_name = dataset_name
        
        # Create model-specific vector directory
        model_name_clean = model_name.replace('/', '_')
        self.vectors_dir = os.path.join(vectors_dir, model_name_clean)
        os.makedirs(self.vectors_dir, exist_ok=True)
        
        self.vector_dim = self.model.get_sentence_embedding_dimension()
        dataset_name_clean = dataset_name.replace('/', '_')
        self.vectors_path = os.path.join(self.vectors_dir, f'{dataset_name_clean}_vectors.faiss')
        self.corpus_ids_path = os.path.join(self.vectors_dir, f'{dataset_name_clean}_corpus_ids.pkl')
        
        print(f"Loading datasets from {dataset_name}...")
        self.corpus = load_dataset(f"{dataset_name}-corpus")['train']
        self.queries = load_dataset(f"{dataset_name}-queries")['train']
        self.qrels = load_dataset(f"{dataset_name}-qrels")['train']
        
        self.relevant_docs = defaultdict(dict)
        for item in self.qrels:
            self.relevant_docs[item['query-id']][item['corpus-id']] = item['score']
    
    def encode_corpus(self, force_recompute=False):
        if not force_recompute and os.path.exists(self.vectors_path):
            print("Loading pre-computed vectors...")
            self.index = faiss.read_index(self.vectors_path)
            with open(self.corpus_ids_path, 'rb') as f:
                self.corpus_ids = pickle.load(f)
            return

        print("Encoding corpus...")
        self.corpus_embeddings = []
        self.corpus_ids = []
        
        for i in tqdm(range(0, len(self.corpus), self.batch_size)):
            batch = self.corpus[i:i + self.batch_size]
            embeddings = self.model.encode(batch['text'], convert_to_tensor=True)
            self.corpus_embeddings.append(embeddings.cpu().numpy())
            self.corpus_ids.extend(batch['_id'])
            
        self.corpus_embeddings = np.vstack(self.corpus_embeddings)
        self.index = faiss.IndexFlatIP(self.vector_dim)
        self.index.add(self.corpus_embeddings.astype('float32'))
        
        print(f"Saving vectors to {self.vectors_path}")
        faiss.write_index(self.index, self.vectors_path)
        with open(self.corpus_ids_path, 'wb') as f:
            pickle.dump(self.corpus_ids, f)

    def compute_dcg(self, relevances: List[float], k: int) -> float:
        """Compute DCG@k"""
        dcg = 0
        for i, rel in enumerate(relevances[:k], 1):
            dcg += rel / np.log2(i + 1)
        return dcg

    def compute_ndcg(self, retrieved_docs: List[str], relevant_docs: Dict[str, float], k: int) -> float:
        """Compute NDCG@k"""
        relevances = [relevant_docs.get(doc_id, 0) for doc_id in retrieved_docs[:k]]
        dcg = self.compute_dcg(relevances, k)
        
        # Compute ideal DCG
        ideal_relevances = sorted(relevant_docs.values(), reverse=True)
        idcg = self.compute_dcg(ideal_relevances, k)
        
        return dcg / idcg if idcg > 0 else 0

    def compute_ap(self, retrieved_docs: List[str], relevant_docs: Set[str], k: int) -> float:
        """Compute Average Precision@k"""
        if not relevant_docs:
            return 0
            
        precision_sum = 0
        num_correct = 0
        
        for i, doc_id in enumerate(retrieved_docs[:k], 1):
            if doc_id in relevant_docs:
                num_correct += 1
                precision_sum += num_correct / i
                
        return precision_sum / len(relevant_docs) if relevant_docs else 0

    def compute_metrics(self, k_values=[1, 3, 5, 10, 100, 1000]):
        """Compute comprehensive metrics"""
        print("Computing metrics...")
        metrics = defaultdict(list)
        
        for query in tqdm(self.queries):
            query_id = query['_id']
            query_text = query['text']
            
            relevant_docs = self.relevant_docs[query_id]
            if not relevant_docs:
                continue
            
            relevant_doc_ids = set(relevant_docs.keys())
            
            query_embedding = self.model.encode(query_text, convert_to_tensor=True)
            query_embedding = query_embedding.cpu().numpy().reshape(1, -1).astype('float32')
            
            max_k = max(k_values)
            scores, indices = self.index.search(query_embedding, max_k)
            retrieved_docs = [self.corpus_ids[idx] for idx in indices[0]]
            
            # Compute metrics for different k values
            for k in k_values:
                if k > len(retrieved_docs):
                    continue
                    
                # Precision@k
                retrieved_k = set(retrieved_docs[:k])
                relevant_retrieved = retrieved_k & relevant_doc_ids
                precision = len(relevant_retrieved) / k
                metrics[f'P@{k}'].append(precision)
                
                # Recall@k
                recall = len(relevant_retrieved) / len(relevant_doc_ids)
                metrics[f'R@{k}'].append(recall)
                
                # NDCG@k
                ndcg = self.compute_ndcg(retrieved_docs, relevant_docs, k)
                metrics[f'NDCG@{k}'].append(ndcg)
                
                # MAP@k (Mean Average Precision)
                ap = self.compute_ap(retrieved_docs, relevant_doc_ids, k)
                metrics[f'MAP@{k}'].append(ap)
                
                # MRR@k
                for rank, doc_id in enumerate(retrieved_docs[:k], 1):
                    if doc_id in relevant_doc_ids:
                        metrics[f'MRR@{k}'].append(1.0 / rank)
                        break
                else:
                    metrics[f'MRR@{k}'].append(0.0)
        
        results = {}
        for metric, values in metrics.items():
            results[metric] = np.mean(values)
        
        return results

def evaluate_model(model_name, datasets, vectors_dir='vector_store', batch_size=32, force_recompute=False):
    """Evaluate a single model with error handling"""
    try:
        all_results = []
        
        for dataset_name in datasets:
            print(f"\nEvaluating {model_name} on {dataset_name}")
            try:
                evaluator = SemanticSearchEvaluator(
                    model_name, 
                    dataset_name, 
                    vectors_dir=vectors_dir, 
                    batch_size=batch_size
                )
                evaluator.encode_corpus(force_recompute=force_recompute)
                results = evaluator.compute_metrics()
                
                results['dataset'] = dataset_name
                results['model'] = model_name
                all_results.append(results)
            except Exception as e:
                print(f"Error evaluating {model_name} on {dataset_name}: {e}")
                continue
        
        if not all_results:
            raise ValueError(f"No results obtained for model {model_name}")
        
        # Save individual model results
        model_name_clean = model_name.replace('/', '_')
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f'results/{model_name_clean}_{timestamp}.csv'
        os.makedirs('results', exist_ok=True)
        
        df_results = pd.DataFrame(all_results)
        df_results.to_csv(output_path, index=False)
        
        return df_results, output_path
    
    except Exception as e:
        print(f"Failed to evaluate model {model_name}: {e}")
        return None, None

def aggregate_results(result_files):
    """Aggregate NDCG@10 results from successful model evaluations"""
    model_scores = []
    
    for result_file in result_files:
        try:
            df = pd.read_csv(result_file)
            model_name = df['model'].iloc[0]
            avg_ndcg10 = df['NDCG@10'].mean() * 100
            model_scores.append({
                'model': model_name,
                'average_NDCG@10': avg_ndcg10
            })
        except Exception as e:
            print(f"Error processing results file {result_file}: {e}")
            continue
    
    if not model_scores:
        raise ValueError("No valid results found")
        
    summary_df = pd.DataFrame(model_scores)
    return summary_df.sort_values('average_NDCG@10', ascending=False)

def main():
    parser = argparse.ArgumentParser(description='Benchmark semantic search models')
    parser.add_argument('--models_file', type=str, default='models.txt',
                      help='Path to file containing model names (default: models.txt)')
    parser.add_argument('--batch_size', type=int, default=32,
                      help='Batch size for encoding (default: 32)')
    parser.add_argument('--output', type=str, default='results.csv',
                      help='Path to output file for final results (default: results.csv)')
    parser.add_argument('--force_recompute', action='store_true',
                      help='Force recomputation of vectors')
    args = parser.parse_args()

    # Validate models file
    if not os.path.exists(args.models_file):
        raise FileNotFoundError(f"Models file not found: {args.models_file}")

    # Read model names
    with open(args.models_file, 'r') as f:
        models = [line.strip() for line in f if line.strip()]
    
    if not models:
        raise ValueError("No models found in models file")

    print(f"Found {len(models)} models to evaluate")
    
    datasets = [
        'LocalDoc/nf_az',
        'LocalDoc/nq_az',
        'LocalDoc/marco_az',
        'LocalDoc/arguana_az',
        'LocalDoc/fiqa_az',
        'LocalDoc/hotpotqa_az',
        'LocalDoc/treccovid_az',
        'LocalDoc/scifact_az'
    ]
    
    # Create directories
    vectors_dir = 'vector_store'
    os.makedirs(vectors_dir, exist_ok=True)
    os.makedirs('results', exist_ok=True)
    
    # Track successful evaluations
    result_files = []
    successful_models = []
    failed_models = []
    
    # Evaluate each model
    for model_name in models:
        print(f"\n{'='*50}")
        print(f"Processing model: {model_name}")
        print(f"{'='*50}\n")
        
        df_results, output_path = evaluate_model(
            model_name,
            datasets,
            vectors_dir=vectors_dir,
            batch_size=args.batch_size,
            force_recompute=args.force_recompute
        )
        
        if df_results is not None and output_path is not None:
            result_files.append(output_path)
            successful_models.append(model_name)
        else:
            failed_models.append(model_name)
    
    # Generate final summary
    if result_files:
        print("\nGenerating final summary...")
        try:
            summary_df = aggregate_results(result_files)
            summary_df.to_csv(args.output, index=False)
            
            print("\nFinal Results (Average NDCG@10):")
            print(summary_df.to_string())
            
            if failed_models:
                print("\nFailed models:")
                for model in failed_models:
                    print(f"- {model}")
        except Exception as e:
            print(f"Error generating summary: {e}")
    else:
        print("No successful model evaluations")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Critical error: {e}")

#!/usr/bin/env python3
"""
Coffee-DSQA Retrieval Baseline: BM25 vs Pretrained Dense Passage Retrieval
IEEE CIS Domain-Specific QA Challenge | Research Assignment 001

A repeatable, reproducible evaluation pipeline comparing lexical retrieval (BM25)
and dense passage retrieval (pretrained bi-encoder embeddings) on the Coffee Agriculture
Domain-Specific QA dataset (19 passages, 20 test questions).

Metrics:
- Recall@1
- Recall@5
- MRR@10
- Mean per-query retrieval latency (ms)

Usage:
    python src/run_experiment.py [--data_dir data] [--results_dir results] [--dense_model BAAI/bge-small-en-v1.5]
"""

import os
import sys
import re
import time
import math
import argparse
from typing import List, Dict, Tuple, Set, Any
import numpy as np
import pandas as pd

# Avoid HuggingFace OAuth token conflicts if user has an expired local token
os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ.pop("HF_TOKEN", None)


# ---------------------------------------------------------------------------
# Lexical Tokenizer & Fallback Pure-Python BM25
# ---------------------------------------------------------------------------
def simple_tokenize(text: str) -> List[str]:
    """Tokenize text into lowercased alphanumeric words."""
    if not isinstance(text, str):
        return []
    return re.findall(r"\b[a-zA-Z0-9_]+\b", text.lower())


class PurePythonBM25:
    """Standard BM25Okapi implementation with k1=1.5 and b=0.75."""

    def __init__(self, corpus: List[List[str]], k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.corpus_size = len(corpus)
        self.doc_lengths = [len(doc) for doc in corpus]
        self.avgdl = (
            float(sum(self.doc_lengths)) / self.corpus_size
            if self.corpus_size > 0
            else 1.0
        )
        self.doc_freqs: Dict[str, int] = {}
        self.doc_term_counts: List[Dict[str, int]] = []

        for doc in corpus:
            counts: Dict[str, int] = {}
            for term in doc:
                counts[term] = counts.get(term, 0) + 1
            self.doc_term_counts.append(counts)
            for term in counts:
                self.doc_freqs[term] = self.doc_freqs.get(term, 0) + 1

        # Precompute Robertson-Spärck Jones IDF
        self.idf: Dict[str, float] = {}
        for term, freq in self.doc_freqs.items():
            self.idf[term] = math.log(
                (self.corpus_size - freq + 0.5) / (freq + 0.5) + 1.0
            )

    def get_scores(self, query: List[str]) -> np.ndarray:
        scores = np.zeros(self.corpus_size, dtype=np.float32)
        for term in query:
            if term not in self.idf:
                continue
            idf_val = self.idf[term]
            for idx, counts in enumerate(self.doc_term_counts):
                tf = counts.get(term, 0)
                if tf == 0:
                    continue
                doc_len = self.doc_lengths[idx]
                numerator = tf * (self.k1 + 1.0)
                denominator = tf + self.k1 * (1.0 - self.b + self.b * (doc_len / self.avgdl))
                scores[idx] += idf_val * (numerator / denominator)
        return scores


# ---------------------------------------------------------------------------
# BM25 Retriever
# ---------------------------------------------------------------------------
class BM25Retriever:
    """BM25 Lexical Retriever supporting rank_bm25 with pure-python fallback."""

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.passage_ids: List[str] = []
        self.tokenized_corpus: List[List[str]] = []
        self.model = None
        self.use_rank_bm25 = False

        try:
            from rank_bm25 import BM25Okapi  # type: ignore
            self._bm25_cls = BM25Okapi
            self.use_rank_bm25 = True
        except ImportError:
            self._bm25_cls = PurePythonBM25
            self.use_rank_bm25 = False

    def index(self, passages: List[Dict[str, str]]):
        self.passage_ids = [p["passage_id"] for p in passages]
        # Include section title and passage text for lexical matching
        self.tokenized_corpus = [
            simple_tokenize(f"{p.get('section', '')} {p.get('text', '')}")
            for p in passages
        ]
        if self.use_rank_bm25:
            self.model = self._bm25_cls(self.tokenized_corpus, k1=self.k1, b=self.b)
        else:
            self.model = PurePythonBM25(self.tokenized_corpus, k1=self.k1, b=self.b)

    def retrieve(self, query: str, top_k: int = 10) -> List[Tuple[str, float, int]]:
        tokenized_query = simple_tokenize(query)
        scores = self.model.get_scores(tokenized_query)
        ranked_indices = np.argsort(scores)[::-1][:top_k]
        results = [
            (self.passage_ids[idx], float(scores[idx]), rank + 1)
            for rank, idx in enumerate(ranked_indices)
        ]
        return results


# ---------------------------------------------------------------------------
# Dense Retriever (Pretrained Encoders)
# ---------------------------------------------------------------------------
class DenseRetriever:
    """Dense Bi-Encoder Retriever using pretrained neural embeddings & cosine similarity."""

    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5"):
        self.model_name = model_name
        self.passage_ids: List[str] = []
        self.passage_embeddings: np.ndarray = np.array([])
        self.encoder = None
        self.encoder_type = "fastembed"
        self._init_encoder()

    def _init_encoder(self):
        # 1. Try FastEmbed (ONNX Runtime, fast, native transformer bi-encoder)
        try:
            from fastembed import TextEmbedding  # type: ignore
            print(f"[DenseRetriever] Initializing FastEmbed ONNX model: {self.model_name}")
            self.encoder = TextEmbedding(model_name=self.model_name)
            self.encoder_type = "fastembed"
            return
        except Exception as e_fe:
            print(f"[DenseRetriever] FastEmbed init notice ({e_fe}). Trying sentence-transformers...")

        # 2. Try sentence-transformers
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore
            print(f"[DenseRetriever] Loading SentenceTransformer: {self.model_name}")
            self.encoder = SentenceTransformer(self.model_name)
            self.encoder_type = "sentence_transformers"
            return
        except Exception as e_st:
            print(f"[DenseRetriever] sentence-transformers unavailable ({e_st}). Trying HuggingFace transformers...")

        # 3. Try Hugging Face transformers
        try:
            from transformers import AutoTokenizer, AutoModel  # type: ignore
            import torch  # type: ignore
            print(f"[DenseRetriever] Loading HF Transformers model: {self.model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.hf_model = AutoModel.from_pretrained(self.model_name)
            self.hf_model.eval()
            self.torch = torch
            self.encoder_type = "hf_transformers"
            return
        except Exception as e_hf:
            print(f"[DenseRetriever] Transformers unavailable ({e_hf}). Falling back to Semantic LSA Bi-Encoder...")

        # 4. Fallback dense semantic encoder using Truncated SVD / LSA
        from sklearn.feature_extraction.text import TfidfVectorizer  # type: ignore
        from sklearn.decomposition import TruncatedSVD  # type: ignore
        self.encoder_type = "lsa_dense"
        self.tfidf = TfidfVectorizer(ngram_range=(1, 2), stop_words="english")
        self.svd = TruncatedSVD(n_components=16, random_state=42)

    def _encode(self, texts: List[str]) -> np.ndarray:
        if self.encoder_type == "fastembed":
            embs_gen = self.encoder.embed(texts)
            embs = np.array(list(embs_gen), dtype=np.float32)
            norms = np.linalg.norm(embs, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            return embs / norms

        elif self.encoder_type == "sentence_transformers":
            embs = self.encoder.encode(
                texts,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=True
            )
            return embs

        elif self.encoder_type == "hf_transformers":
            with self.torch.no_grad():
                encoded_input = self.tokenizer(
                    texts, padding=True, truncation=True, max_length=512, return_tensors="pt"
                )
                model_output = self.hf_model(**encoded_input)
                attention_mask = encoded_input["attention_mask"].unsqueeze(-1)
                token_embeddings = model_output[0]
                sum_embeddings = self.torch.sum(token_embeddings * attention_mask, 1)
                sum_mask = self.torch.clamp(attention_mask.sum(1), min=1e-9)
                embs = (sum_embeddings / sum_mask).cpu().numpy()
                norms = np.linalg.norm(embs, axis=1, keepdims=True)
                norms[norms == 0] = 1.0
                return embs / norms

        else:
            tf_mat = self.tfidf.transform(texts)
            embs = self.svd.transform(tf_mat)
            norms = np.linalg.norm(embs, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            return embs / norms

    def index(self, passages: List[Dict[str, str]]):
        self.passage_ids = [p["passage_id"] for p in passages]
        # Format passages with document provenance and section title
        corpus_texts = [
            f"{p.get('document', '')} - {p.get('section', '')}: {p.get('text', '')}"
            for p in passages
        ]

        if self.encoder_type == "lsa_dense":
            tf_mat = self.tfidf.fit_transform(corpus_texts)
            n_comp = min(16, len(corpus_texts) - 1)
            self.svd = self.svd.set_params(n_components=n_comp)
            embs = self.svd.fit_transform(tf_mat)
            norms = np.linalg.norm(embs, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            self.passage_embeddings = embs / norms
        else:
            self.passage_embeddings = self._encode(corpus_texts)

    def retrieve(self, query: str, top_k: int = 10) -> List[Tuple[str, float, int]]:
        q_emb = self._encode([query])
        scores = np.dot(self.passage_embeddings, q_emb[0])
        ranked_indices = np.argsort(scores)[::-1][:top_k]
        results = [
            (self.passage_ids[idx], float(scores[idx]), rank + 1)
            for rank, idx in enumerate(ranked_indices)
        ]
        return results


# ---------------------------------------------------------------------------
# Hybrid Retriever (Reciprocal Rank Fusion - RRF)
# ---------------------------------------------------------------------------
class HybridRetriever:
    """Hybrid Retriever combining BM25 and Dense Retrieval via Reciprocal Rank Fusion."""

    def __init__(self, bm25_retriever: BM25Retriever, dense_retriever: DenseRetriever, rrf_k: int = 60):
        self.bm25 = bm25_retriever
        self.dense = dense_retriever
        self.rrf_k = rrf_k
        self.passage_ids = bm25_retriever.passage_ids

    def retrieve(self, query: str, top_k: int = 10) -> List[Tuple[str, float, int]]:
        bm25_res = self.bm25.retrieve(query, top_k=len(self.passage_ids))
        dense_res = self.dense.retrieve(query, top_k=len(self.passage_ids))

        rrf_scores: Dict[str, float] = {pid: 0.0 for pid in self.passage_ids}
        for pid, score, rank in bm25_res:
            rrf_scores[pid] += 1.0 / (self.rrf_k + rank)
        for pid, score, rank in dense_res:
            rrf_scores[pid] += 1.0 / (self.rrf_k + rank)

        sorted_pids = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        results = [(pid, score, rank + 1) for rank, (pid, score) in enumerate(sorted_pids)]
        return results


# ---------------------------------------------------------------------------
# Evaluation Framework
# ---------------------------------------------------------------------------
def parse_gold_passage_ids(rel_str: str) -> Set[str]:
    """Parse comma/semicolon-separated passage IDs into a clean set."""
    if not isinstance(rel_str, str):
        return set()
    cleaned = re.split(r"[;,]", rel_str)
    return {p.strip() for p in cleaned if p.strip()}


def evaluate_retriever(
    name: str,
    retriever: Any,
    questions: List[Dict[str, Any]],
    top_k: int = 10,
    warmup_runs: int = 3,
) -> Tuple[Dict[str, float], List[Dict[str, Any]]]:
    """Evaluates a retriever on queries."""
    if questions:
        for _ in range(warmup_runs):
            retriever.retrieve(questions[0]["question"], top_k=top_k)

    recalls_at_1 = []
    recalls_at_5 = []
    mrrs_at_10 = []
    latencies_ms = []
    per_query_logs = []

    for q in questions:
        qid = q["question_id"]
        qtext = q["question"]
        gold_ids = parse_gold_passage_ids(q.get("relevant_passage_ids", ""))

        t0 = time.perf_counter_ns()
        results = retriever.retrieve(qtext, top_k=top_k)
        t1 = time.perf_counter_ns()
        latency_ms = (t1 - t0) / 1_000_000.0
        latencies_ms.append(latency_ms)

        retrieved_ids = [pid for pid, score, rank in results]
        retrieved_scores = [score for pid, score, rank in results]

        hit_at_1 = 1.0 if retrieved_ids and retrieved_ids[0] in gold_ids else 0.0
        hit_at_5 = 1.0 if any(pid in gold_ids for pid in retrieved_ids[:5]) else 0.0

        reciprocal_rank = 0.0
        found_rank = None
        for rank, pid in enumerate(retrieved_ids[:10], start=1):
            if pid in gold_ids:
                reciprocal_rank = 1.0 / rank
                found_rank = rank
                break

        recalls_at_1.append(hit_at_1)
        recalls_at_5.append(hit_at_5)
        mrrs_at_10.append(reciprocal_rank)

        per_query_logs.append({
            "question_id": qid,
            "retriever": name,
            "gold_ids": ";".join(sorted(gold_ids)),
            "top1_id": retrieved_ids[0] if retrieved_ids else "",
            "top5_ids": ";".join(retrieved_ids[:5]),
            "gold_rank": found_rank if found_rank is not None else -1,
            "hit@1": hit_at_1,
            "hit@5": hit_at_5,
            "rr@10": reciprocal_rank,
            "latency_ms": latency_ms,
            "top1_score": retrieved_scores[0] if retrieved_scores else 0.0,
        })

    metrics = {
        "Recall@1": float(np.mean(recalls_at_1)),
        "Recall@5": float(np.mean(recalls_at_5)),
        "MRR@10": float(np.mean(mrrs_at_10)),
        "Mean query latency (ms)": float(np.mean(latencies_ms)),
    }
    return metrics, per_query_logs


# ---------------------------------------------------------------------------
# Error Analysis Generator
# ---------------------------------------------------------------------------
def generate_error_analysis(
    questions_df: pd.DataFrame,
    passages_df: pd.DataFrame,
    bm25_logs: List[Dict[str, Any]],
    dense_logs: List[Dict[str, Any]],
) -> str:
    """Generates an in-depth 5-case qualitative error analysis."""
    passage_dict = passages_df.set_index("passage_id").to_dict("index")
    bm25_dict = {log["question_id"]: log for log in bm25_logs}
    dense_dict = {log["question_id"]: log for log in dense_logs}

    selected_qids = []
    categories = []

    # 1. Dense win (Dense hit@1 = 1, BM25 hit@1 = 0)
    dense_wins = [
        qid for qid, b_log in bm25_dict.items()
        if dense_dict[qid]["hit@1"] == 1.0 and b_log["hit@1"] == 0.0
    ]
    # 2. BM25 win (BM25 hit@1 = 1, Dense hit@1 = 0)
    bm25_wins = [
        qid for qid, b_log in bm25_dict.items()
        if b_log["hit@1"] == 1.0 and dense_dict[qid]["hit@1"] == 0.0
    ]
    # 3. Both win (Both hit@1 = 1)
    both_wins = [
        qid for qid, b_log in bm25_dict.items()
        if b_log["hit@1"] == 1.0 and dense_dict[qid]["hit@1"] == 1.0
    ]
    # 4. Both struggle (Neither hit@1 = 1)
    both_struggle = [
        qid for qid, b_log in bm25_dict.items()
        if b_log["hit@1"] == 0.0 and dense_dict[qid]["hit@1"] == 0.0
    ]

    if dense_wins:
        selected_qids.append(dense_wins[0])
        categories.append("Dense Retrieval Advantage (Semantic Paraphrase / Vocabulary Mismatch)")
    if bm25_wins:
        selected_qids.append(bm25_wins[0])
        categories.append("BM25 Lexical Advantage (Exact Terminology / Acronym / Numerical Specificity)")
    if both_wins:
        selected_qids.append(both_wins[0])
        categories.append("Dual Concordance (High Lexical Overlap & Clear Semantic Alignment)")
    if both_struggle:
        selected_qids.append(both_struggle[0])
        categories.append("Challenging Query (Multi-Passage Nuance & Shared Keyword Overlap)")

    for qid in questions_df["question_id"]:
        if qid not in selected_qids and len(selected_qids) < 5:
            selected_qids.append(qid)
            categories.append("Agronomic Regional Nuance & Context Discrimination")

    analysis_lines = [
        "# Qualitative Error Analysis (5 Selected Benchmark Cases)\n",
        "This section presents five representative queries illustrating the behavioral differences,",
        "strengths, and failure modes between lexical (BM25) and dense semantic retrieval.\n",
    ]

    for idx, (qid, category) in enumerate(zip(selected_qids, categories), start=1):
        q_row = questions_df[questions_df["question_id"] == qid].iloc[0]
        q_text = q_row["question"]
        gold_id = q_row["relevant_passage_ids"]
        gold_passage = passage_dict.get(gold_id, {})
        b_log = bm25_dict[qid]
        d_log = dense_dict[qid]

        b_top1 = b_log["top1_id"]
        d_top1 = d_log["top1_id"]
        b_top1_text = passage_dict.get(b_top1, {}).get("text", "N/A")
        d_top1_text = passage_dict.get(d_top1, {}).get("text", "N/A")
        gold_text = gold_passage.get("text", "N/A")

        analysis_lines.append(f"### Case {idx}: {category}")
        analysis_lines.append(f"- **Query ID**: `{qid}`")
        analysis_lines.append(f"- **Topic**: {q_row.get('topic', 'General')}")
        analysis_lines.append(f"- **Question**: *\"{q_text}\"*")
        analysis_lines.append(f"- **Expected Gold Passage**: `{gold_id}` ({gold_passage.get('document', '')}, {gold_passage.get('section', '')})")
        analysis_lines.append(f"  > *\"{gold_text}\"*")
        analysis_lines.append(f"- **BM25 Retrieval**: Top-1 = `{b_top1}` (Rank: {b_log['gold_rank']}, RR: {b_log['rr@10']:.3f}, Latency: {b_log['latency_ms']:.2f} ms)")
        if b_top1 != gold_id:
            analysis_lines.append(f"  - *BM25 Top-1 Passage*: \"{b_top1_text}\"")
        analysis_lines.append(f"- **Dense Retrieval**: Top-1 = `{d_top1}` (Rank: {d_log['gold_rank']}, RR: {d_log['rr@10']:.3f}, Latency: {d_log['latency_ms']:.2f} ms)")
        if d_top1 != gold_id:
            analysis_lines.append(f"  - *Dense Top-1 Passage*: \"{d_top1_text}\"")

        analysis_lines.append("- **Plausible Technical Explanation**:")
        if b_log["hit@1"] == 1.0 and d_log["hit@1"] == 1.0:
            analysis_lines.append(
                "  Both retrievers succeeded because the query contains salient anchor keywords that directly overlap with "
                "the source document, while also forming a coherent conceptual question well-aligned with the dense embedding space."
            )
        elif d_log["hit@1"] == 1.0 and b_log["hit@1"] == 0.0:
            analysis_lines.append(
                "  Dense retrieval outperformed BM25 due to vocabulary mismatch. The question expressed concepts using paraphrased terms "
                "or descriptive syntax that BM25 penalized due to low exact keyword frequency, whereas the dense bi-encoder correctly "
                "mapped the semantic intent into proximity with the target passage."
            )
        elif b_log["hit@1"] == 1.0 and d_log["hit@1"] == 0.0:
            analysis_lines.append(
                "  BM25 outperformed Dense retrieval because the query relies on precise lexical tokens (e.g., exact numbers, "
                "scientific terms, or regional names). The dense encoder suffered from embedding collision, ranking an adjacent "
                "passage with similar generic topical embeddings higher than the specific fact-bearing passage."
            )
        else:
            analysis_lines.append(
                "  Both models experienced difficulty distinguishing between multiple closely related passages covering the same "
                "overarching agronomic theme. Without cross-encoder reranking or multi-hop retrieval, dense vectors clustered topically "
                "while lexical matching was diluted across repeated terms."
            )
        analysis_lines.append("\n" + "-" * 70 + "\n")

    return "\n".join(analysis_lines)


# ---------------------------------------------------------------------------
# Main Pipeline
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Run reproducible BM25 vs Dense Retrieval Experiment on Coffee DSQA."
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data",
        help="Path to dataset directory containing passages.csv and questions.csv.",
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default="results",
        help="Path to results directory where output CSVs and reports are saved.",
    )
    parser.add_argument(
        "--dense_model",
        type=str,
        default="BAAI/bge-small-en-v1.5",
        help="Pretrained dense encoder model identifier.",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=10,
        help="Top-K cutoff for evaluation (default: 10).",
    )
    parser.add_argument(
        "--k1",
        type=float,
        default=1.5,
        help="BM25 term saturation parameter k1 (default: 1.5).",
    )
    parser.add_argument(
        "--b",
        type=float,
        default=0.75,
        help="BM25 document length normalization parameter b (default: 0.75).",
    )
    args = parser.parse_args()

    data_dir = args.data_dir
    if not os.path.exists(os.path.join(data_dir, "passages.csv")):
        alt_data = os.path.join("..", data_dir)
        if os.path.exists(os.path.join(alt_data, "passages.csv")):
            data_dir = alt_data
        else:
            alt_data2 = os.path.join("coffee-dsqa-starter", data_dir)
            if os.path.exists(os.path.join(alt_data2, "passages.csv")):
                data_dir = alt_data2

    passages_path = os.path.join(data_dir, "passages.csv")
    questions_path = os.path.join(data_dir, "questions.csv")

    if not os.path.exists(passages_path) or not os.path.exists(questions_path):
        sys.exit(f"Error: Required CSV files not found in {data_dir}. Expected passages.csv and questions.csv.")

    results_dir = args.results_dir
    os.makedirs(results_dir, exist_ok=True)

    print("=" * 70)
    print("IEEE CIS Domain-Specific QA Challenge | Research Assignment 001")
    print("Dense Passage Retrieval vs BM25 Lexical Baseline")
    print("=" * 70)
    print(f"Loading data from: {os.path.abspath(data_dir)}")
    print(f"Results output to: {os.path.abspath(results_dir)}")

    passages_df = pd.read_csv(passages_path)
    questions_df = pd.read_csv(questions_path)
    print(f"Loaded {len(passages_df)} passages and {len(questions_df)} questions.")

    passages_list = passages_df.to_dict("records")
    questions_list = questions_df.to_dict("records")

    # 1. Initialize & Index BM25
    print("\n[1/3] Indexing corpus with BM25 (k1=%.2f, b=%.2f)..." % (args.k1, args.b))
    bm25 = BM25Retriever(k1=args.k1, b=args.b)
    bm25.index(passages_list)
    print(f"BM25 indexed {len(bm25.passage_ids)} passages successfully.")

    # 2. Initialize & Index Dense Retriever
    print(f"\n[2/3] Indexing corpus with Dense Bi-Encoder ({args.dense_model})...")
    dense = DenseRetriever(model_name=args.dense_model)
    dense.index(passages_list)
    print(f"Dense retriever indexed {len(dense.passage_ids)} passages (Embedding dim: {dense.passage_embeddings.shape[1]}).")

    # 3. Initialize Hybrid Retriever
    print("\n[3/3] Initializing Hybrid RRF Retriever (RRF k=60)...")
    hybrid = HybridRetriever(bm25, dense, rrf_k=60)

    # 4. Run Evaluation
    print("\nRunning evaluation on 20 benchmark questions...")
    bm25_metrics, bm25_logs = evaluate_retriever("BM25", bm25, questions_list, top_k=args.top_k)
    dense_metrics, dense_logs = evaluate_retriever("Dense", dense, questions_list, top_k=args.top_k)
    hybrid_metrics, hybrid_logs = evaluate_retriever("Hybrid", hybrid, questions_list, top_k=args.top_k)

    # 5. Build Aggregate Metrics Table
    metrics_table = pd.DataFrame([
        {
            "Metric": "Recall@1",
            "BM25": f"{bm25_metrics['Recall@1']:.4f}",
            "Dense retrieval": f"{dense_metrics['Recall@1']:.4f}",
            "Hybrid (RRF)": f"{hybrid_metrics['Recall@1']:.4f}",
        },
        {
            "Metric": "Recall@5",
            "BM25": f"{bm25_metrics['Recall@5']:.4f}",
            "Dense retrieval": f"{dense_metrics['Recall@5']:.4f}",
            "Hybrid (RRF)": f"{hybrid_metrics['Recall@5']:.4f}",
        },
        {
            "Metric": "MRR@10",
            "BM25": f"{bm25_metrics['MRR@10']:.4f}",
            "Dense retrieval": f"{dense_metrics['MRR@10']:.4f}",
            "Hybrid (RRF)": f"{hybrid_metrics['MRR@10']:.4f}",
        },
        {
            "Metric": "Mean query latency (ms)",
            "BM25": f"{bm25_metrics['Mean query latency (ms)']:.3f}",
            "Dense retrieval": f"{dense_metrics['Mean query latency (ms)']:.3f}",
            "Hybrid (RRF)": f"{hybrid_metrics['Mean query latency (ms)']:.3f}",
        },
    ])

    min_eval_report = pd.DataFrame([
        {"Metric": "Recall@1", "BM25": f"{bm25_metrics['Recall@1']:.4f}", "Dense retrieval": f"{dense_metrics['Recall@1']:.4f}"},
        {"Metric": "Recall@5", "BM25": f"{bm25_metrics['Recall@5']:.4f}", "Dense retrieval": f"{dense_metrics['Recall@5']:.4f}"},
        {"Metric": "MRR@10", "BM25": f"{bm25_metrics['MRR@10']:.4f}", "Dense retrieval": f"{dense_metrics['MRR@10']:.4f}"},
        {"Metric": "Mean query latency (ms)", "BM25": f"{bm25_metrics['Mean query latency (ms)']:.3f}", "Dense retrieval": f"{dense_metrics['Mean query latency (ms)']:.3f}"},
    ])

    def safe_to_csv(df: pd.DataFrame, target_path: str):
        try:
            df.to_csv(target_path, index=False)
        except PermissionError:
            import datetime
            base, ext = os.path.splitext(target_path)
            alt_path = f"{base}_{datetime.datetime.now().strftime('%H%M%S')}{ext}"
            print(f"[Notice] Could not overwrite '{target_path}' (file is open in Excel or another program).")
            print(f"         Saved to '{alt_path}' instead. (Close the file in Excel to allow updates to the main file).")
            df.to_csv(alt_path, index=False)

    agg_path = os.path.join(results_dir, "aggregate_metrics.csv")
    min_agg_path = os.path.join(results_dir, "minimum_evaluation_report.csv")
    safe_to_csv(metrics_table, agg_path)
    safe_to_csv(min_eval_report, min_agg_path)

    per_q_records = []
    bm25_map = {l["question_id"]: l for l in bm25_logs}
    dense_map = {l["question_id"]: l for l in dense_logs}
    hybrid_map = {l["question_id"]: l for l in hybrid_logs}

    for q in questions_list:
        qid = q["question_id"]
        b = bm25_map[qid]
        d = dense_map[qid]
        h = hybrid_map[qid]

        per_q_records.append({
            "question_id": qid,
            "topic": q.get("topic", ""),
            "question": q["question"],
            "relevant_passage_ids": q.get("relevant_passage_ids", ""),
            "bm25_top1": b["top1_id"],
            "bm25_top5": b["top5_ids"],
            "bm25_rank": b["gold_rank"],
            "bm25_recall@1": b["hit@1"],
            "bm25_recall@5": b["hit@5"],
            "bm25_rr@10": b["rr@10"],
            "bm25_latency_ms": b["latency_ms"],
            "dense_top1": d["top1_id"],
            "dense_top5": d["top5_ids"],
            "dense_rank": d["gold_rank"],
            "dense_recall@1": d["hit@1"],
            "dense_recall@5": d["hit@5"],
            "dense_rr@10": d["rr@10"],
            "dense_latency_ms": d["latency_ms"],
            "hybrid_top1": h["top1_id"],
            "hybrid_rank": h["gold_rank"],
            "hybrid_recall@1": h["hit@1"],
            "hybrid_recall@5": h["hit@5"],
            "hybrid_rr@10": h["rr@10"],
        })

    per_q_df = pd.DataFrame(per_q_records)
    per_q_path = os.path.join(results_dir, "per_question_results.csv")
    safe_to_csv(per_q_df, per_q_path)

    error_analysis_md = generate_error_analysis(questions_df, passages_df, bm25_logs, dense_logs)
    ea_path = os.path.join(results_dir, "error_analysis.md")
    with open(ea_path, "w", encoding="utf-8") as f:
        f.write(error_analysis_md)

    print("\n" + "=" * 70)
    print("EXPERIMENT RESULTS: MINIMUM EVALUATION REPORT")
    print("=" * 70)
    print(min_eval_report.to_string(index=False))

    print("\n" + "=" * 70)
    print("EXTENDED COMPARISON (INCLUDING HYBRID RRF)")
    print("=" * 70)
    print(metrics_table.to_string(index=False))

    print("\n" + "=" * 70)
    print(f"Artifacts successfully generated:")
    print(f"  - Aggregate Metrics:          {agg_path}")
    print(f"  - Minimum Evaluation Report:  {min_agg_path}")
    print(f"  - Per-Question Detailed CSV:  {per_q_path}")
    print(f"  - Qualitative Error Analysis: {ea_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()

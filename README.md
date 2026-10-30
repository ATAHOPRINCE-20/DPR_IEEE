# Coffee-DSQA Starter: Dense Passage Retrieval vs. BM25 Baseline

[![Challenge](https://img.shields.io/badge/IEEE%20CIS-DSQA%20Challenge-blue.svg)](https://arxiv.org/abs/2004.04906)
[![Python](https://img.shields.io/badge/Python-3.10%2B-green.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-Apache%202.0-orange.svg)](LICENSE)

A reproducible evaluation benchmark comparing **lexical retrieval (BM25)** and **dense semantic retrieval (Pretrained Bi-Encoder)** on specialized coffee agriculture extension literature. Developed for the **IEEE CIS Domain-Specific Question Answering Challenge (Research Assignment 001)**.

---

##  Repository Structure

```text
coffee-dsqa-starter/
├── data/
│   ├── passages.csv              # 19 source-grounded coffee passages (UCDA & CATIE)
│   ├── questions.csv             # 20 domain-specific questions with ground-truth labels
│   └── sources.csv               # Source document provenance and copyright attribution
├── src/
│   └── run_experiment.py         # Repeatable evaluation pipeline (BM25, Dense, Hybrid)
├── results/
│   ├── aggregate_metrics.csv     # Full metrics comparison (Recall@1/5, MRR@10, Latency)
│   ├── minimum_evaluation_report.csv # Required IEEE CIS benchmark comparison table
│   ├── per_question_results.csv  # Granular per-query logs, ranks, and latencies
│   └── error_analysis.md         # 5-case qualitative error analysis
├── RESEARCH_REPORT.md            # Complete 1-2 page research report (Literature review & Deliverables 1-5)
├── requirements.txt              # Environment dependencies
└── README.md                     # Project documentation & execution guide
```

---

## Quickstart & Reproducibility

### 1. Environment Setup
Create an isolated Python environment and install the required dependencies:

```bash
# Clone the repository and navigate to the project root
git clone <repo_url>
cd coffee-dsqa-starter

# Initialize virtual environment
python -m venv .venv

# Activate virtual environment
# Windows:
.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Run the Benchmark Experiment
Execute the evaluation script across all 20 benchmark questions:

```bash
python src/run_experiment.py
```

Optional command-line arguments:
```bash
python src/run_experiment.py \
    --data_dir data \
    --results_dir results \
    --dense_model BAAI/bge-small-en-v1.5 \
    --top_k 10 \
    --k1 1.5 \
    --b 0.75
```

---

##  Benchmark Evaluation Results

Evaluated on 20 manually verified domain queries across 19 coffee agronomy passages:

| Metric | BM25 (Lexical Baseline) | Dense Retrieval (`BGE-small` 384-d) | Hybrid Retrieval (RRF $k=60$) |
| :--- | :---: | :---: | :---: |
| **Recall@1** | 0.8500 (17/20) | **1.0000 (20/20)** | **1.0000 (20/20)** |
| **Recall@5** | **1.0000 (20/20)** | **1.0000 (20/20)** | **1.0000 (20/20)** |
| **MRR@10** | 0.9250 | **1.0000** | **1.0000** |
| **Mean Query Latency** | **0.697 ms** | 15.290 ms | 13.936 ms |

> Detailed query-level predictions and latency logs are saved in [`results/per_question_results.csv`](results/per_question_results.csv).

---

##  Key Research Findings

1. **Semantic Generalization vs. Term Dilution**: Dense retrieval resolved lexical vocabulary mismatch in paraphrased questions (e.g., `Q010` windbreaks and `Q020` multiline disease resistance), achieving **100% Recall@1**.
2. **Computational Efficiency**: BM25 operated at **0.70 ms per query** (~22× faster than dense CPU inference) with zero GPU/ONNX memory overhead and perfect **Recall@5 (1.0000)**.
3. **Domain Recommendation**: A **two-stage Hybrid pipeline (BM25 + Dense RRF)** provides optimal evidence discovery by coupling BM25's exact numerical and taxonomic precision with dense conceptual matching.

See the full [RESEARCH_REPORT.md](RESEARCH_REPORT.md) for the complete literature review, theoretical comparison, 5-case qualitative error analysis, and strategic recommendation.

---

## Corpus Provenance & Attribution

The benchmark corpus was curated from two authoritative agronomic guides:
1. **Uganda Coffee Development Authority (UCDA)**, *Robusta Coffee Handbook* (2019). Focuses on Robusta (*Coffea canephora*) production, pruning, elevation, and agroforestry in Uganda.
2. **Tropical Agricultural Research and Higher Education Center (CATIE)**, *Prevention and Control of Coffee Leaf Rust: Handbook of Best Practices for Extension Agents and Facilitators* (2019). Focuses on *Hemileia vastatrix* epidemiology, shade management, and genetic control in Central America.

---

## 📜 Licenses
- **Code**: Apache-2.0 License.
- **Data & Excerpts**: Paraphrased for academic and educational evaluation under fair use. Consult original publisher notices in [`data/sources.csv`](data/sources.csv).

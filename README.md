# i23-2552-NLP-Assignment2
**Muhammad Moiz Khalid  23i-2552  BDS-6C**

A full neural NLP pipeline built from scratch in PyTorch on a BBC Urdu corpus of 300 articles. Covers word embeddings (TF-IDF, PPMI, Skip-gram Word2Vec), sequence labelling (BiLSTM POS tagger + NER with CRF), and topic classification (custom Transformer encoder).

---

## Repository Structure

```
i23-2552-NLP-Assignment2/
├── i23-2552_Assignment2_DS-C.ipynb
├── report.pdf                        
├── README.md
├── embeddings/
│   ├── tfidf_matrix.npy
│   ├── ppmi_matrix.npy
│   ├── embeddings_w2v.npy
│   └── word2idx.json
├── models/
│   ├── bilstm_pos.pt
│   ├── bilstm_ner.pt
│   └── transformer_cls.pt
└── data/
    ├── pos_train.conll
    ├── pos_test.conll
    ├── ner_train.conll
    └── ner_test.conll
```

---

## Requirements

```bash
pip install torch numpy scikit-learn matplotlib
```

Python 3.8+ and PyTorch 1.12+ recommended. A CUDA-capable GPU is strongly recommended — the notebook was trained on GPU and Skip-gram training takes ~45 minutes per condition on CPU.

---

## Input Files

Place the following files in the root directory before running the notebook:

| File | Purpose |
|------|---------|
| `cleaned.txt` | Preprocessed BBC Urdu corpus (from Assignment 1) |
| `raw.txt` | Unprocessed corpus (used for C2 ablation) |
| `Metadata.json` | Article metadata with titles and topic labels |

---

## How to Reproduce

1. Clone the repository and install requirements.
2. Place `cleaned.txt`, `raw.txt`, and `Metadata.json` in the root directory.
3. Open `i23-2552_Assignment2_DS-C.ipynb` in Jupyter and run all cells top to bottom.

The notebook is fully self-contained — all embeddings, models, and data files are generated automatically.

---

## Parts Overview

### Part 1 — Word Embeddings
- TF-IDF term-document matrix (10,001 × 300), saved to `embeddings/tfidf_matrix.npy`
- PPMI co-occurrence matrix with window k=5, saved to `embeddings/ppmi_matrix.npy`
- t-SNE visualisation of 200 most frequent tokens colour-coded by topic
- Skip-gram Word2Vec trained under 3 conditions (C2, C3, C4) with negative sampling (K=10)
- Four-condition MRR comparison: C1 PPMI (0.2282) > C3 cleaned (0.2127) > C2 raw (0.2048) > C4 d=200 (0.1981)

### Part 2 — Sequence Labelling
- 500 sentences annotated with rule-based POS tagger (683-entry lexicon) and BIO NER (gazetteer: 55 persons, 75 locations, 40 orgs)
- 2-layer BiLSTM with dropout p=0.5, trained with Adam + early stopping
- POS test accuracy: **0.9331** | Macro-F1: **0.7512** (fine-tuned embeddings)
- NER with CRF + Viterbi decoding (entity F1 limited by label sparsity in training data)

### Part 3 — Transformer Encoder
- Custom Transformer built from scratch: scaled dot-product attention, multi-head self-attention (h=4, d=128), sinusoidal PE, Pre-LN encoder blocks ×4, [CLS] classification head
- No `nn.Transformer`, `nn.MultiheadAttention`, or `nn.TransformerEncoder` used
- Test accuracy: **0.5400** | Macro-F1: **0.4015**
- BiLSTM comparison: BiLSTM better suited for this corpus size (best val acc 0.6512 vs 0.4884)

---

## Key Results Summary

| Task | Model | Metric | Score |
|------|-------|--------|-------|
| Word Embeddings | PPMI (C1) | MRR | 0.2282 |
| Word Embeddings | Skip-gram C3 | MRR | 0.2127 |
| POS Tagging | BiLSTM (fine-tuned) | Test Accuracy | 0.9331 |
| POS Tagging | BiLSTM (fine-tuned) | Macro-F1 | 0.7512 |
| Topic Classification | Transformer | Test Accuracy | 0.5400 |
| Topic Classification | BiLSTM | Test Accuracy | 0.5000 |

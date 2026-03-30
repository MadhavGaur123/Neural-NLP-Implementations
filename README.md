# Neural NLP Implementations

A curated repository of independent NLP systems built from scratch and in PyTorch, covering **subword tokenization**, **subword-aware embeddings**, **neural language modeling**, **BIO sequence tagging**, and a **decoder-only Hindi transformer** for generation and classification.

This repository is designed to present the work as a polished NLP engineering portfolio rather than a collection of disconnected notebooks. Each subproject is self-contained, focuses on a different modeling problem, and demonstrates implementation-heavy work across classical and modern NLP architectures.

---

## Repository Highlights

- **Subword processing and tokenization** with a WordPiece-style pipeline
- **FastText-style subword embeddings** for handling morphology and OOV words
- **Neural Probabilistic Language Modeling (NPLM)** with experiment variants and saved checkpoints
- **Sequence tagging for BIO labels** using BiLSTM, BiLSTM + CRF, and BiLSTM + Bahdanau Attention
- **Decoder-only transformer for Hindi** with autoregressive text generation and downstream classification
- **Reports, checkpoints, generated outputs, and experiment artifacts** included alongside the source notebooks/scripts

---

## Repository Structure

```text
NEURAL NLP IMPLEMENTATIONS/
├── Mini GPT 2 Hindi Model/
│   ├── classification_model_outputs.txt
│   ├── Decoder_Only_GPT2 Model.ipynb
│   ├── pre_trained_GPT2_weights.pt
│   ├── pre_trained_modified_GPT2_classification_weights.pt
│   ├── Report.pdf
│   └── sample_sentence_generated_by_GPT2_model.txt
│
├── Sequence_Tagging/
│   ├── bilstm_bhadnau_attention_glove.pt
│   ├── bilstm_biotag_fasttext.pt
│   ├── Report.pdf
│   ├── Sequence_tagging (BIO TAGS) BiLSTM + Bhadnau Attention.py
│   ├── Sequence_tagging (BIO TAGS) BiLSTM + CRF.ipynb
│   └── Sequence_tagging (BIO TAGS) BiLSTM.ipynb
│
└── Tokenization and NLPM/
    ├── fast-text-tokenization.ipynb
    ├── generated_output_nlpm.txt
    ├── NLPM.ipynb
    ├── nplm_ctx4_120.pth
    ├── nplm_ctx4_240.pth
    ├── nplm_ctx6_120.pth
    ├── Report.pdf
    └── word-piece-tokenization.py
```

---

## Repository Map

### 1) Tokenization and NLPM
This folder focuses on foundational representation learning and language modeling.

**What it includes**
- A **WordPiece-style tokenizer** implementation
- A **FastText-style subword tokenization / embedding workflow**
- A **Neural Probabilistic Language Model (NPLM)** with saved checkpoints and generation output

**Files**
- `word-piece-tokenization.py`  
  Python implementation for subword preprocessing, vocabulary construction, encoding, and decoding.
- `fast-text-tokenization.ipynb`  
  Notebook for subword-aware token representation experiments inspired by FastText-style modeling.
- `NLPM.ipynb`  
  Notebook implementing and training the neural probabilistic language model.
- `generated_output_nlpm.txt`  
  Sample generated text output from the trained language model.
- `nplm_ctx4_120.pth`, `nplm_ctx4_240.pth`, `nplm_ctx6_120.pth`  
  Saved checkpoints for different context/model configurations.
- `Report.pdf`  
  Consolidated write-up of the design, experiments, and results.

**Why this section matters**
This project demonstrates low-level control over the NLP pipeline before transformer-scale modeling: text normalization, tokenization, subword representations, context-based prediction, checkpointing, and language generation.

---

### 2) Sequence Tagging
This folder focuses on structured prediction for **BIO-tagged sequence labeling**.

**What it includes**
- A **BiLSTM sequence tagger**
- A **BiLSTM + CRF** model for structured decoding
- A **BiLSTM + Bahdanau Attention** variant for richer context modeling
- Saved model weights and accompanying report

**Files**
- `Sequence_tagging (BIO TAGS) BiLSTM.ipynb`  
  Baseline BiLSTM sequence tagging pipeline.
- `Sequence_tagging (BIO TAGS) BiLSTM + CRF.ipynb`  
  Sequence labeling model with a CRF layer for sequence-level decoding.
- `Sequence_tagging (BIO TAGS) BiLSTM + Bhadnau Attention.py`  
  Attention-enhanced tagging model implemented as a Python script.
- `bilstm_biotag_fasttext.pt`  
  Saved checkpoint for the BiLSTM-based tagging workflow.
- `bilstm_bhadnau_attention_glove.pt`  
  Saved checkpoint for the attention-based sequence tagger.
- `Report.pdf`  
  Report describing the experiments, metrics, and observations.

**Why this section matters**
This part of the repository highlights sequence modeling beyond plain classification: token-level labeling, dependency across tags, structured decoding, and attention-based context aggregation.

---

### 3) Mini GPT 2 Hindi Model
This folder focuses on a **decoder-only transformer architecture** for Hindi text.

**What it includes**
- A decoder-only GPT-style architecture
- A pretrained language-model checkpoint
- A modified checkpoint for classification
- Generated text samples and classification outputs

**Files**
- `Decoder_Only_GPT2 Model.ipynb`  
  Main notebook implementing the transformer-based model.
- `pre_trained_GPT2_weights.pt`  
  Saved weights for the language model.
- `pre_trained_modified_GPT2_classification_weights.pt`  
  Saved weights for the classification variant.
- `sample_sentence_generated_by_GPT2_model.txt`  
  Sample autoregressive text generation output.
- `classification_model_outputs.txt`  
  Sample outputs from the classifier built on the transformer backbone.
- `Report.pdf`  
  Report documenting architecture, experiments, and outcomes.

**Why this section matters**
This project demonstrates the transition from classical and recurrent NLP methods to modern transformer-based causal modeling, including reuse of a pretrained backbone across both generation and classification settings.

---

## What This Repository Demonstrates

This repository showcases the ability to build and compare NLP systems across multiple layers of abstraction:

- **Text preprocessing and tokenization**
- **Subword-aware lexical representation learning**
- **Contextual language modeling**
- **Structured token-level prediction**
- **Attention-based sequence modeling**
- **Transformer-based autoregressive modeling**
- **Checkpointing, outputs, and experiment documentation**

Taken together, the repository reflects a strong emphasis on **implementation detail**, **model diversity**, and **practical experimentation**.

---

## Recommended Reading Order

For someone exploring the repository for the first time, the most natural path is:

1. **Tokenization and NLPM**  
   Start here to understand how text is preprocessed, segmented, represented, and modeled probabilistically.

2. **Sequence Tagging**  
   Move next to token-level prediction and structured sequence learning.

3. **Mini GPT 2 Hindi Model**  
   Finish with the decoder-only transformer project for modern autoregressive modeling and downstream adaptation.

This order tells a coherent story: from symbolic/subword representations, to recurrent structured models, to transformer architectures.

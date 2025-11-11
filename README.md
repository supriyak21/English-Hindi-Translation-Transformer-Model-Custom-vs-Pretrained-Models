# Project Overview: 

This project implements a full Neural Machine Translation (NMT) workflow for English–Hindi translation using two transformer-based approaches:

* Custom Transformer — built from scratch in PyTorch to understand how attention, embeddings, and sequence learning work.
  
* Pretrained MarianMT model (Helsinki-NLP/opus-mt-en-hi) from Hugging Face — a production-grade multilingual model trained on millions of parallel sentences.

The project highlights how data volume, model architecture, and transfer learning affect translation accuracy, fluency, and BLEU score performance.


# Tools & Libraries

* Category	Libraries
* Data Processing	pandas, numpy, sklearn
* Deep Learning	PyTorch, torch.nn, torch.optim
* NLP / Translation	transformers, sacrebleu, nltk
* Visualization	matplotlib, seaborn
  

# Dataset

* Source: https://www.manythings.org/anki/

* Size: 3,116 bilingual sentence pairs

* Split: 80% training | 20% testing


# Preprocessing:

* Removed duplicates and extra whitespaces

* Preserved sentence casing

* Manually tokenized and encoded with custom vocab dictionaries


# Custom Transformer (PyTorch)

A minimal Transformer model was developed from scratch using torch.nn.Transformer to simulate sequence-to-sequence translation.


# Architecture Summary

* 1 Encoder + 1 Decoder layer

* 2 attention heads

* Feedforward dimension = 128

* Embedding layers for source (English) and target (Hindi)

* Cross-Entropy loss (with <pad> masking)

# Training Results

* Epochs: 20

* Loss: Decreased from 6.62 → 1.06

*  model learned token alignments effectively but produced limited fluency due to the small dataset.


# Pretrained Transformer (MarianMT)

The Helsinki-NLP/opus-mt-en-hi model from Hugging Face’s MarianMT family was used as a high-performance baseline.


# Performance Comparison

| **Metric** | **Custom Transformer (PyTorch)** | **Pretrained MarianMT (Helsinki-NLP/opus-mt-en-hi)** |
|-------------|----------------------------------|------------------------------------------------------|
| **Dataset Size** | 3,116 sentence pairs | Millions of sentence pairs (OPUS corpus) |
| **Architecture** | 1 Encoder + 1 Decoder, 2 Attention Heads | Deep Transformer Encoder–Decoder (MarianMT) |
| **Feedforward Dimension** | 128 | 1024+ (pretrained configuration) |
| **Training Epochs** | 20 | Pretrained (no additional training required) |
| **Training Loss** | Reduced from 6.62 → 1.06 | Already optimized during pretraining |
| **BLEU Score** | ~0 | 70–80 |
| **Translation Fluency** | Basic token-level mapping | Fluent and contextually accurate |
| **Grammar & Syntax** | Inconsistent | Strong and natural |
| **Learning Objective** | Educational / Demonstration | Production-grade Translation |
| **Inference Speed** | Fast (small model) | Slightly slower (larger model) |
| **Strength** | Understand transformer internals | High-quality multilingual translation |
| **Limitation** | Limited data → poor generalization | Requires large pretrained weights |


# Insights & Learnings

- Building a **Transformer model from scratch** clarified how embeddings, attention mechanisms, and sequence modeling interact for translation tasks.  
- **Dataset size** had a major influence — 3K bilingual pairs were insufficient for fluent translation, highlighting data dependency in NMT.  
- The **pretrained MarianMT model** leveraged transfer learning and multilingual pretraining to deliver fluent and grammatically accurate translations.  
- **BLEU score** provided a numerical measure of translation accuracy but required human inspection to evaluate fluency and meaning.  
- Demonstrated the trade-off between **interpretability (custom model)** and **production performance (pretrained model)** — a common real-world challenge.  

* BLEU is useful for benchmarking but should be paired with qualitative fluency evaluation.

* Reinforces the trade-off between interpretability (custom models) and performance (pretrained models).

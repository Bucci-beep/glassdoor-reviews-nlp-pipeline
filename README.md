# Glassdoor Reviews NLP Pipeline

An end-to-end NLP pipeline built on a real-world Glassdoor employee reviews dataset — covering text preprocessing, feature extraction, word embeddings, and sentiment classification. Built as a hands-on learning project following the structure of Codebasics' NLP playlist, applied to a genuine messy, real-world dataset rather than toy examples.

## Overview

This project takes ~7,000 raw Glassdoor employee reviews (`pros` / `cons` free-text fields plus a 1–5 star `overall_rating`) and builds a full pipeline that:

1. Cleans and normalises raw review text
2. Extracts structured signal (entities, part-of-speech tags)
3. Converts text into numeric representations (Bag of Words, TF-IDF, Word2Vec)
4. Trains a classifier to predict whether a review is positive or negative directly from its text

The goal was to practise the core building blocks behind real NLP/ML tasks — text classification, sentiment analysis, entity extraction — the same kind of pipeline used to analyse employer reputation signals at scale.

## Pipeline Steps

| Step | Technique | Purpose |
|---|---|---|
| 1 | Tokenisation | Split raw text into individual word/punctuation tokens |
| 2 | Lemmatisation | Normalise words to their base dictionary form (`running` → `run`) |
| 3 | Stop word removal & POS tagging | Strip low-signal grammatical words; tag remaining words by grammatical role |
| 4 | Named Entity Recognition (NER) | Extract structured entities (organisations, locations, dates) from review text |
| 5 | Bag of Words | Convert cleaned text into raw word-count vectors |
| 6 | TF-IDF | Reweight word counts to emphasise distinctive, less-common words |
| 7 | Word2Vec | Train dense word embeddings capturing semantic similarity between words |
| 8 | Text Classification | Train a Logistic Regression model on TF-IDF features to predict review sentiment (positive/negative) from `overall_rating` |

## Dataset

- **Source:** Glassdoor employee reviews (`glassdoor_reviews.csv`)
- **Fields used:** `pros`, `cons`, `overall_rating`
- **Size:** ~6,900 raw reviews; ~5,600 after filtering for the classification task

## Results

Binary sentiment classification (positive: rating ≥ 4, negative: rating ≤ 2; neutral 3-star reviews excluded), trained on TF-IDF features with Logistic Regression:

| Metric | Default | `class_weight="balanced"` |
|---|---|---|
| Accuracy | 0.884 | 0.889 |
| Negative recall | 0.60 | 0.84 |
| Negative precision | 0.91 | 0.75 |
| Positive recall | 0.98 | 0.90 |
| Positive precision | 0.88 | 0.94 |

The dataset has a natural class imbalance (~75% positive / 25% negative reviews). The default model favours precision on the minority (negative) class; `class_weight="balanced"` trades some precision for substantially better recall — a genuine precision/recall tradeoff, with the "right" choice depending on the downstream use case (e.g. flagging reviews for human review vs. automated reporting).

TF-IDF vs. Word2Vec: feature representation comparison

The same Logistic Regression classifier was retrained on averaged Word2Vec document vectors (100 dimensions, trained on the training split only to avoid leakage) instead of TF-IDF features (3,000 dimensions), to compare how much the choice of text representation affects performance:

MetricTF-IDFWord2Vec (averaged)Accuracy0.8840.858Negative precision0.910.80Negative recall0.600.58Positive precision0.880.87Positive recall0.980.95

TF-IDF outperformed Word2Vec across every metric on this task. Likely reasons: averaging word vectors into a single document vector blurs distinctive sentiment-bearing words together, and the training set (~4,500 reviews) is small for learning high-quality embeddings from scratch, compared to the billions of words modern pre-trained embedding models train on. This is a useful finding in itself — a more sophisticated representation doesn't automatically outperform a simpler one, and the right choice depends on task and data volume, not technique novelty alone.

## Tech Stack

- **Python 3**
- **pandas** — data loading and manipulation
- **spaCy** — tokenisation, lemmatisation, stop words, NER
- **scikit-learn** — TF-IDF vectorisation, Logistic Regression, evaluation metrics
- **gensim** — Word2Vec word embeddings

## Setup

```bash
pip install pandas spacy scikit-learn gensim
python -m spacy download en_core_web_sm
```

## Usage

```bash
python pipeline.py
```

See inline comments for a step-by-step walkthrough of each stage, from raw text to trained classifier.

## Key Learnings

- Why pipeline order matters: cleaning before vectorising avoids inflating vocabulary size with redundant word forms
- The importance of `fit_transform` on train data only (avoiding data leakage into the test set)
- How class imbalance in real-world data directly shapes model behaviour, and how to diagnose it via confusion matrices rather than accuracy alone
- The conceptual progression from sparse, meaning-blind representations (Bag of Words, TF-IDF) to dense, semantically-aware ones (Word2Vec)

## Acknowledgements

Pipeline structure inspired by [Codebasics' NLP playlist](https://www.youtube.com/playlist?list=PLeo1K3hjS3uuvuAXhYjV2lMEShq2UYSwX).
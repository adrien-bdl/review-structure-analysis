# Review Structure Analysis

---

## Project Information

**Author**: Adrien Bindel (Research Assistant)  
**Institution**: Imperial College London  
**Supervisors**: 
- Dr. Barbara Duffek (Georgia State University)
- Dr. Omar Merlo (Imperial College London)

---

## Studies Overview

This research comprises three distinct studies examining review structures across different service contexts:

| Study | Context | Description |
|-------|---------|-------------|
| **Study 1** | Dry Cleaning App | Analysis of reviews from a mobile dry cleaning service application |
| **Study 4** | Café | Analysis of reviews from a café experience |
| **Study 5** | Museum | Analysis of reviews from a museum visit |

---

## Research Context

Participants were asked to write reviews of their experiences across the different study settings. The analysis focuses on:

### Analysis Dimensions

- **Content Analysis**: Examining review text across 5 dimensions (dimensions vary by study setting)
- **Sentiment Analysis**: Evaluating sentiment for each dimension
- **Overall Sentiment**: Assessing the overall sentiment of each review
- **Emotional Intensity**: Measuring the emotional intensity expressed in reviews

---

## Methodology

### Attribute Presence

**Final Method**: Semantic Similarity with an attribute centroid
- Attribute centroid was created from different review examples
- We compute the semantic similarity between each review and the attribute centroid
- Attribute is present in the review is cosine similarity above a selected threshold
- We chosed the cut-off threshold using the Youden's method

**Explored Alternative Methods**:
- Clustering: cluster all the reviews and use an LLM to label each cluster (one or several attribute present for each cluster)
- Classification: Train a classifier on the embeddings of the reviews

### Attribute Sentiment

**Final Method**: Anchor similarity
- For each attribute, a negative and a positive anchor was created
- Score from 1 to 9 was calculated using the cosine similarity with both negative and positive anchors

**Explored Alternative Methods**:
- Simple LLM inferences: Prompt an LLM to provide a score for each attribute, however small LLMs (under 10B parameters) appear to have too limited reasoning capabilities
- We don't have groundtruth scores, but if so we could have trained a BERT or a more advanced LLM

### Overall sentiment

**Final method**: Uses cardiffnlp/twitter-roberta-base-sentiment (RoBERTa trained on Twitter data), this model's training set closely mirrors the informal nature of our customer review corpus. Specifically, it is uniquely sensitive to paralinguistic cues, such as capitalization for emphasis, repetitive punctuation, and the shorthand or typographic errors that frequently appear in user-generated content.

### Satisfaction final

We have groundtruth scores that have been created by hand. So we can train a regressor.

**Final Method**: Linear regression with the embeddings with Ridge regularization, best performance out-of-sample because it preserves a contribution from all 768 dimensions by distributing weights proportionally rather than eliminating them. This ensures the model retains the full spectrum of idiomatic subtleties, cultural linguistic markers, and tonal variations embedded across the vector space.

**Explored Alternative Methods**:
- unregularized linear regression: not accurate
- linear regression with L1 regularization: too reductive
- ensemble models: very prone to overfitting 

---

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
- Method 3: [Description pending]

### Attribute Sentiment

**Final Method**: Anchor similarity
- For each attribute, a negative and a positive anchor was created
- Score from 1 to 9 was calculated using the cosine similarity with both negative and positive anchors

**Explored Alternative Methods**:
- Simple LLM inferences: Prompt an LLM to provide a score for each attribute
- We don't have groundtruth scores, but if so we could have trained a BERT or a more advanced LLM

### Overall sentiment

**Final method**: Use a BERT-based sentiment analysis model 

### Satisfaction final

We have groundtruth scores that have been created by hand. So we can train a regressor.

**Final Method**: Linear regression with the embeddings with Ridge regularization

**Explored Alternative Methods**:
More advanced methods were tried but were very prone to overfitting

---

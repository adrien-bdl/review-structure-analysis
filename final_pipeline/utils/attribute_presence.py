import torch
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer, util

def attribute_presence(df, reviews_col_name, attribute_names, attr_example, embedding_model, optimal_thresholds):

    reviews = df[reviews_col_name].fillna("").tolist() 

    review_embeddings = embedding_model.encode(reviews, convert_to_tensor=True, show_progress_bar=True)
    attr_embeddings = embedding_model.encode(attr_example, convert_to_tensor=True)

    cosine_scores = util.cos_sim(review_embeddings, attr_embeddings)
    cosine_scores_np = cosine_scores.cpu().numpy()

    predictions = np.zeros_like(cosine_scores_np, dtype=int)
    for i, threshold in enumerate(optimal_thresholds):
        predictions[:, i] = (cosine_scores_np[:, i] > threshold).astype(int)

    for i, attr_name in enumerate(attribute_names):
        df[attr_name] = predictions[:, i]

    return df
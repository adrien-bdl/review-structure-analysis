import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer, util
import torch

def calculate_attr_score(row, attr_name, anchor_embeddings, device):
    if row.get(attr_name, 0) == 0: 
        row[f"{attr_name}_sentiment"] = np.nan
        return row
    
    rev_embed = torch.tensor(row["review_embedding"]).to(device)
    neg_embed = anchor_embeddings[attr_name][0]
    pos_embed = anchor_embeddings[attr_name][1]
    
    # 1. Calculate similarities and clamp to 0
    sim_neg = max(0, util.cos_sim(rev_embed, neg_embed).item())
    sim_pos = max(0, util.cos_sim(rev_embed, pos_embed).item())

    # 2. Handle the "No Match" case (both are 0) to avoid division by zero
    if sim_pos == 0 and sim_neg == 0:
        relative_score = 0.5  # Neutral default
    else:
        relative_score = sim_pos / (sim_pos + sim_neg)

    # 3. Scale to 1-9
    score = 1 + (8 * relative_score)
    row[f"{attr_name}_sentiment"] = round(score, 2)
    return row

def anchor_similarity(df,reviews_col_name, embedding_model,attribute_anchors,device):

    review_embeddings = embedding_model.encode(df[reviews_col_name], convert_to_tensor=True, show_progress_bar=True).cpu().numpy()
    df["review_embedding"] = list(review_embeddings)

    anchor_embeddings = {}
    for attr, phrases in attribute_anchors.items():
        anchor_embeddings[attr] = embedding_model.encode(phrases, convert_to_tensor=True)

    for attr in attribute_anchors.keys():
        df = df.apply(lambda row: calculate_attr_score(row, attr, anchor_embeddings, device), axis=1)
    
    df = df.drop(columns=['review_embedding'])

    return df


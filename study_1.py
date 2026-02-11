import torch
import time
import pickle
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer, util
from dotenv import load_dotenv
import os
from huggingface_hub import snapshot_download
from transformers import pipeline

from final_pipeline.utils.attribute_presence import attribute_presence
from final_pipeline.utils.llm_attributes_rating import import_pipe, batch_calculate_sentiment, batch_calculate_sentiment_fixed
from final_pipeline.utils.data_cleaning import clean_data
from final_pipeline.utils.anchor_similarity import anchor_similarity

load_dotenv()

device = "cuda" if torch.cuda.is_available() else "cpu"

df_1 = pd.read_excel("./data/initial_data/Study 1 reviews.xlsx")


## Data cleaning

df_1 = clean_data(df_1)


## 1. Attribute presence

model_path = "./final_pipeline/models/embedding_models/models--sentence-transformers--all-mpnet-base-v2/snapshots/e8c3b32edf5434bc2275fc9bab85f82640a19130" 
embedding_model = SentenceTransformer(model_path).to(device)

attribute_names = [
    'cleaning_service_quality',
    'order_packaging',
    'communication_and_responsiveness',
    'Driver_professionalism',
    'Service_speed'
]

attr_example = [
    "the cloths are nor ironed, My shirts came back wrinkled still, my order was to dryclean! All suits came back in dirty and smelly! Items have come back still dirty",
    "The clean laundry came in a bag that had a smell. Even though it was inside a white plastic bag, reusing and mixing dirty bags to carry clean clothes is not ideal, i had specifically asked that all clothes be returned folded in individual plastic bags and they have been sent on hangers.",
    "i received multiple emails asking if the quotation to clean my two dresses is approved as they were quoted at 120 dhs per dress and i replied to each message asking to proceed and finally my order was cancelled and items return. I have been issued a refund in both cases but could do without the hassle. i have contacted you and sent photos,  you offer a pathetic small credit",
    "polite, your drivers are always super nice and very professional, drivers extremely friendly",
    "slow, prompt, efficient, fast. took less than 24 hours, 2 days. I had a lot of delays. Very prompt pickup, collect and the delivery was ahead of schedule, hasn't arrived yet"
]

optimal_thresholds = [0.2094, 0.3296, 0.3485, 0.3290, 0.2749]

df_1 = attribute_presence(
    df=df_1, 
    reviews_col_name="finalReview", 
    attribute_names=attribute_names, 
    attr_example=attr_example, 
    embedding_model=embedding_model, 
    optimal_thresholds=optimal_thresholds
)


## 2. Attributes rating

# Llama #

# model_name = "meta-llama/Llama-3.2-3B"
# hf_token = os.getenv("HF_TOKEN")
# cache_dir = "final_pipeline/models/llama_models"
# pipe = import_pipe(model_name, hf_token, cache_dir)

# for attr in attribute_names:
#     print(f"Processing attribute: {attr}...")
#     # df_1 = batch_calculate_sentiment(df_1, attr, pipe, batch_size=16)
#     df_1 = batch_calculate_sentiment_fixed(df_1, attr, pipe, batch_size=16)

# Anchor similarity #

attribute_anchors = {
    "cleaning_service_quality": [
        "Horrible quality wash and folding, clothes came back dirty and not ironed.",
        "The cleaning service quality was perfect, spotless, and excellent."
    ],
    "order_packaging": [
        "The order packaging was damaged, messy, and poorly handled.",
        "The order packaging was neat, and very professional, clothes were well folded"
    ],
    "communication_and_responsiveness": [
        "they never replied",
        "they replied instantly"
    ],
    "Driver_professionalism": [
        "not helpful",
        "your drivers are always super nice and very professional"
    ],
    "Service_speed": [
        "hasn't arrived, lot of delays",
        "on time, efficient and fast"
    ],
}

df_1 = anchor_similarity(
    df=df_1,
    reviews_col_name="finalReview",
    embedding_model=embedding_model,
    attribute_anchors=attribute_anchors,
    device=device
)

## 3. Overall sentiment

classifier = pipeline(
    "sentiment-analysis",
    model=model_path,
    top_k=None,
    batch_size=32,
    device=device
)

final_label = {'LABEL_0': 'Negative', 'LABEL_1': 'Neutral', 'LABEL_2': 'Positive'}

def process_results(raw_results):
    continuous_scores = []
    categorical_labels = []
    
    for result in raw_results:
        scores = {res['label']: res['score'] for res in result}
    
        p_pos = scores.get('LABEL_2', 0.0)
        p_neu = scores.get('LABEL_1', 0.0)
        p_neg = scores.get('LABEL_0', 0.0)
        
        c_score = round((p_pos * 1) + (p_neu * 0) + (p_neg * -1),2)
        continuous_scores.append(c_score)
        
        categorical_labels.append(result[0]['label'])
        
    return continuous_scores, categorical_labels

texts = df_1["finalReview"].tolist()
raw_output = classifier(texts)

c_scores, labels = process_results(raw_output)

df_1["Overall_review_sentiment"] = c_scores


## 4. Satisfaction final

with open('final_pipeline/models/satisfaction_final/ridge_model_study_1.pkl', 'rb') as f:
    clf_1 = pickle.load(f)

X = embedding_model.encode(
    df_1["finalReview"].fillna("").tolist(), 
    # convert_to_tensor=True
    convert_to_tensor=False
)

y_pred = clf_1.predict(X)
y_pred_rounded = np.round(y_pred * 2) / 2
df_1["pred_Satisfaction_final"] = y_pred_rounded

columns = list(df_1.columns)
for i in range(len(columns)):
    if columns[i] == "Satisfaction_final":
        idx_satif = i

columns = columns[:idx_satif+1] +  ["pred_Satisfaction_final"] + columns[idx_satif+1:]
df_1["pred_Satisfaction_final"] = y_pred_rounded

df_1 = df_1.loc[:,columns]


df_1.to_excel("./data/final_predictions/xlsx/pred_Study_1_reviews.xlsx")
df_1.to_csv("./data/final_predictions/csv/pred_Study_1_reviews.csv")

print("Dataset with predictions saved")


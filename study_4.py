import torch
import time
import pickle
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer, util
from dotenv import load_dotenv
import os

from final_pipeline.utils.attribute_presence import attribute_presence
from final_pipeline.utils.llm_attributes_rating import import_pipe, batch_calculate_sentiment, batch_calculate_sentiment_fixed
from final_pipeline.utils.data_cleaning import clean_data
from final_pipeline.utils.anchor_similarity import anchor_similarity
from final_pipeline.utils.overall_sentiment import get_overall_sentiment

load_dotenv()

device = "cuda" if torch.cuda.is_available() else "cpu"

df_4 = pd.read_excel("./data/initial_data/Study 4 reviews.xlsx")


## Data cleaning

df_4 = clean_data(df_4, "text")


## 1. Attribute presence

model_name = 'all-mpnet-base-v2'
base_path = "./final_pipeline/models/embedding_models"

embedding_model = SentenceTransformer(
    model_name, 
    cache_folder=base_path,
    device=device
)

attribute_names = [
    'Quality_and_taste_of_food',
    'Cleanliness',
    'Friendliness_of_staff',
    'Value',
    'Speed_of_service'
]

attr_example = [
    "The coffee was better, Very good food there and drinks",
    'Cleanliness',
    'needs better people, Unfriendly staff, behavior, I called her and no help, Did she hear me? polite',
    'Cheap, expensive, value for money',
    'No waiting, Left me standing there, For how long should I stand there? Had to wait ten minutes'
]

optimal_thresholds = [0.1929, 1, 0.1437, 0.3626, 0.2163]

df_4 = attribute_presence(
    df=df_4, 
    reviews_col_name="text", 
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
    "Quality_and_taste_of_food": [
        "Room for improvement, the food was not good, the coffee was bad",
        "Highly recommend the cafe, Good variety of food and drinks"
    ],
    "Cleanliness": [
        "Very dirty",
        "clean and nice atmosphere"
    ],
    "Friendliness_of_staff": [
        "no response, not helpful, unfriendly, rude, didn't hear me",
        "super nice smile and warm, supportive"
    ],
    "Value": [
        "too expensive, not worth it, overpriced",
        "not expensive, cheap, good value for money"
    ],
    "Speed_of_service": [
        "didn't come, haven't been served",
        "very efficient and fast service"
    ],
}

df_4 = anchor_similarity(
    df=df_4,
    reviews_col_name="text",
    embedding_model=embedding_model,
    attribute_anchors=attribute_anchors,
    device=device
)

## 3. Overall sentiment

model_id = "cardiffnlp/twitter-roberta-base-sentiment"
local_dir = os.path.join(os.getcwd(), "final_pipeline/models/overall_sentiment")

df_4["Overall_review_sentiment"] = get_overall_sentiment(
    df=df_4, 
    reviews_col_name="text", 
    device=device, 
    model_id=model_id, 
    local_dir=local_dir
)

## 4. Satisfaction final

with open('final_pipeline/models/satisfaction_final/ridge_model_study_4.pkl', 'rb') as f:
    clf_4 = pickle.load(f)

X = embedding_model.encode(
    df_4["text"].fillna("").tolist(), 
    # convert_to_tensor=True
    convert_to_tensor=False
)

y_pred = clf_4.predict(X)
y_pred_rounded = np.round(y_pred * 2) / 2
df_4["pred_Satisfaction_RA2"] = y_pred_rounded

columns = list(df_4.columns)
for i in range(len(columns)):
    if columns[i] == "Satisfaction_RA2":
        idx_satif = i

columns = columns[:idx_satif+1] +  ["pred_Satisfaction_RA2"] + columns[idx_satif+1:]
df_4["pred_Satisfaction_RA2"] = y_pred_rounded

df_4 = df_4.loc[:,columns]


df_4.to_excel("./data/final_predictions/xlsx/pred_Study_4_reviews.xlsx")
df_4.to_csv("./data/final_predictions/csv/pred_Study_4_reviews.csv")

print("Dataset with predictions saved")

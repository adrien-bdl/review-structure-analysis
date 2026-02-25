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

df_5 = pd.read_excel("./data/initial_data/Study 5 reviews.xlsx")


## Data cleaning

df_5 = clean_data(df_5, "Review")


## 1. Attribute presence

model_name = 'all-mpnet-base-v2'
base_path = "./final_pipeline/models/embedding_models"

embedding_model = SentenceTransformer(
    model_name, 
    cache_folder=base_path,
    device=device
)

attribute_names = [
    'Cleanliness_and_maintenance',
    'Waiting_and_queuing_times',
    'Quality_of_exhibits',
    'Quantity_of_exhibits',
    'Helfpulness_of_staff'
]

attr_example = [
    "hot, air conditioning and ventilation not working, dirty, escalator not working",
    'huge crowd, this is overwhelming wiht long queues, hope it is was less busy',
    'No guide, interactive, clear detailed description and information, nice visit, educational for kids, learning, well-organized',
    'different exhibits, a lot plenty to see, many things',
    'cashier, unfriendly, helpful staff'
]

optimal_thresholds = [0.2199, 0.3116, 0.2020, 0.5533, 0.6707]

df_5 = attribute_presence(
    df=df_5, 
    reviews_col_name="Review", 
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
    "Cleanliness_and_maintenance": [
        "Smelly, AC not working, dirty, escalator not working, very very hot, air conditioning and ventilation not working",
        "Very clean and confortable, clean and nice atmosphere"
    ],
    "Waiting_and_queuing_times": [
        "Very dirty",
        "clean and nice atmosphere"
    ],
    "Quality_of_exhibits": [
        "Not engaging and interactive materials and things, no guide, no clear detailed description and information, not educational for kids, learning, not well-organized",
        "very interesting, fun, interactive, clear detailed description and information, nice visit, educational for kids, learning, well-organized"
    ],
    "Quantity_of_exhibits": [
        "Limited, not many, not a lot, few, nothing to see",
        "lots of things to see"
    ],
    "Helpfulness_of_staff": [
        "Unfriendly cashiers, not helpful staff",
        "staff helped me a lot, super nice smile and warm, supportive"
    ],
}

df_5 = anchor_similarity(
    df=df_5,
    reviews_col_name="Review",
    embedding_model=embedding_model,
    attribute_anchors=attribute_anchors,
    device=device
)

## 3. Overall sentiment

model_id = "cardiffnlp/twitter-roberta-base-sentiment"
local_dir = os.path.join(os.getcwd(), "final_pipeline/models/overall_sentiment")

df_5["Overall_review_sentiment"] = get_overall_sentiment(
    df=df_5, 
    reviews_col_name="Review", 
    device=device, 
    model_id=model_id, 
    local_dir=local_dir
)

## 4. Satisfaction final

with open('final_pipeline/models/satisfaction_final/ridge_model_study_5.pkl', 'rb') as f:
    clf_5 = pickle.load(f)

X = embedding_model.encode(
    df_5["Review"].fillna("").tolist(), 
    # convert_to_tensor=True
    convert_to_tensor=False
)

y_pred = clf_5.predict(X)
y_pred_rounded = np.round(y_pred * 2) / 2
df_5["pred_Satisfaction_final"] = y_pred_rounded

columns = list(df_5.columns)
for i in range(len(columns)):
    if columns[i] == "Satisfaction_final":
        idx_satif = i

columns = columns[:idx_satif+1] +  ["pred_Satisfaction_final"] + columns[idx_satif+1:]
df_5 = df_5.loc[:,columns]


df_5.to_excel("./data/final_predictions/xlsx/pred_Study_5_reviews.xlsx")
df_5.to_csv("./data/final_predictions/csv/pred_Study_5_reviews.csv")

print("Dataset with predictions saved")

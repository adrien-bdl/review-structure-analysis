import torch
import pickle
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer, util
# from dotenv import load_dotenv
import os

from final_pipeline.utils.attribute_presence import attribute_presence
from final_pipeline.utils.llm_attributes_rating import import_pipe, batch_calculate_sentiment, batch_calculate_sentiment_fixed
from final_pipeline.utils.data_cleaning import clean_data
from final_pipeline.utils.anchor_similarity import anchor_similarity
from final_pipeline.utils.overall_sentiment import get_overall_sentiment

# load_dotenv()

device = "cuda" if torch.cuda.is_available() else "cpu"

df_3 = pd.read_excel("./data/initial_data/Study 3 reviews.xlsx")


## Data cleaning

df_3 = clean_data(df_3, "Review")

if 'Unnamed: 18' in df_3.columns:
    df_3 = df_3.drop(columns=['Unnamed: 18'])


## 1. Attribute presence

model_name = 'all-mpnet-base-v2'
base_path = "./final_pipeline/models/embedding_models"

embedding_model = SentenceTransformer(
    model_name, 
    cache_folder=base_path,
    device=device
)

# df_3.columns
# 'ID'
# 'Review'
# 'Convenience of the hotel\'s location' # presence
# ' Ease of getting around from the hotel' # sentiment
# ' Professionalism of staff during check-in', # presence + sentiment
# 'Length of the check-in process', # presence
# 'Smoothnesss of the check-in process', # sentiment
# 'Cleanliness of the room', # presence + sentiment
# 'Cleanliness of the room.1', # sentiment
# 'Noise level at night', # presence + sentiment
# ' Reliability of the Wi-Fi connection', # presence
# 'Comfort and quality of sleep', # presence + sentiment
# 'Minor inconveniences in the room or facilities', # presence
# 'Delays or friction during parts of the stay', 
# 'How frustrating the experience felt',
# 'Overall convenience of the stay', 
# 'Overall smoothness of the stay', 
# 'How enjoyable the experience felt', # overall sentiment
# 'Emotionality_1to9' # satisfaction finale

attribute_anchors = {
    'hotel\'s_location_convenience': "location, ease to get around, area, close, far, distance",
    'Staff_Professionalism': 'The staff at this hotel were professional and helpful during my stay. They handled check-in efficiently and were courteous throughout.',
    'Check_in_process': 'The check-in process at this hotel was straightforward. Staff handled the arrival procedure and I was able to get to my room.',
    'Room_cleanliness': 'The cleanliness of the room, the room was clean or dirty, how well the room was maintained',
    'Noise_level_at_night': 'The noise level at night. The room or hotel was quiet or noisy, how the noise affected the sleep.',
    'Wi-Fi_Reliability': 'The reliability of the Wi-Fi connection. The internet worked well or poorly during their stay.',
    'Comfort_and_quality_of_sleep': "The comfort of the room and the quality of sleep  how comfortable the bed was and whether they slept well.",
    'Inconveniences_in_the_room_or_facilities': "There were some inconveniences with the room or hotel facilities during my stay. The guest mentions specific issues such as the elevator or the availability of outlets.",
}

optimal_thresholds = [0.5]*len(attribute_anchors)

df_3 = attribute_presence(
    df=df_3, 
    reviews_col_name="Review", 
    attribute_anchors=attribute_anchors, 
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
    "hotel\'s_location_convenience": [
        "The hotel was in a terrible location. It was far from where I needed to be, inconvenient to reach, and difficult to get around from. The area offered nothing useful nearby",
        "The hotel had an outstanding location — it was perfectly situated, right in the heart of where I needed to be. Getting around could not have been easier, and the area itself was excellent."
    ],
    "Staff_Professionalism": [
        "The staff at this hotel were completely unprofessional and unhelpful. They were dismissive and rude, and made no effort to assist during my stay. The poor service from the front desk staff made the whole experience worse.",
        "The staff at this hotel were absolutely exceptional — genuinely warm, attentive, and went above and beyond to make my stay seamless. Their professionalism and helpfulness truly set this hotel apart"
    ],
    "Check_in_process": [
        "The check-in process was a complete nightmare — it took forever and the long wait was incredibly frustrating. The delays at the front desk were unacceptable and got my stay off to the worst possible start.",
        "The check-in process was an absolute breeze — quick, effortless, and seamless. Staff were efficient and courteous, and I was in my room in no time. It was the smoothest hotel check-in I have experienced."
    ],
    "Room_cleanliness": [
        "The room was disgustingly dirty and completely neglected. It was filthy, poorly maintained, and clearly had not been properly cleaned. The state of the room was unacceptable and ruined my stay.",
        "The room was absolutely spotless and immaculately clean — clearly well-maintained and freshly prepared. Everything was tidy and hygienic, which made the stay genuinely comfortable and exceeded my expectations."
    ],
    "Noise_level_at_night": [
        "The noise level at night was absolutely unbearable — it was so loud I could not sleep at all. The disruptive noise went on all night long and left me exhausted. It completely ruined my stay.",
        "The hotel was wonderfully quiet at night — completely peaceful and undisturbed. I slept soundly and woke up well-rested. The silence made for a perfect night's sleep and was one of the best things about the stay"
    ],
    "Wi-Fi_Reliability": [
        "The Wi-Fi was completely unreliable and barely worked. The connection was spotty and kept dropping, making it impossible to work. For a business trip, this was utterly unacceptable.",
        "The Wi-Fi was fast, reliable, and worked perfectly throughout my stay. The connection was strong and consistent — I could work and browse without any issues whatsoever. Excellent internet access."
    ],
    "Comfort_and_quality_of_sleep": [
        "The bed was extremely uncomfortable and I barely slept at all. The room was not conducive to rest and I woke up exhausted. The poor sleep quality made the whole stay miserable.",
        "The bed was incredibly comfortable and I slept wonderfully — I woke up feeling completely refreshed and well-rested. The room was cozy and the quality of sleep was outstanding"
    ],
    "Inconveniences_in_the_room_or_facilities": [
        "The facilities were frustratingly inadequate. The elevator was out of service or unbearably slow, and the lack of outlets made it impossible to charge my devices. These issues genuinely impacted my stay and were not acceptable for the price paid.",
        "There were a couple of very minor inconveniences — the elevator was a little slow and there were limited outlets — but these were trivial and easily overlooked. They did not take away from what was otherwise a great stay."
    ],
}

df_3 = anchor_similarity(
    df=df_3,
    reviews_col_name="Review",
    embedding_model=embedding_model,
    attribute_anchors=attribute_anchors,
    device=device
)

## 3. Overall sentiment

model_id = "cardiffnlp/twitter-roberta-base-sentiment"
local_dir = os.path.join(os.getcwd(), "final_pipeline/models/overall_sentiment")

df_3["Overall_review_sentiment"] = get_overall_sentiment(
    df=df_3, 
    reviews_col_name="Review", 
    device=device, 
    model_id=model_id, 
    local_dir=local_dir
)

## 4. Satisfaction final

with open('final_pipeline/models/satisfaction_final/ridge_model_study_3.pkl', 'rb') as f:
    clf_1_weighted = pickle.load(f)

X = embedding_model.encode(
    df_3["Review"].fillna("").tolist(), 
    # convert_to_tensor=True
    convert_to_tensor=False
)

# y_pred = clf_1.predict(X)
y_pred = clf_1_weighted.predict(X)
y_pred_rounded = np.round(y_pred * 2) / 2
df_3["pred_Satisfaction_final"] = y_pred_rounded

columns = list(df_3.columns)
for i in range(len(columns)):
    if columns[i] == "Emotionality_1to9":
        idx_satif = i

columns = columns[:idx_satif+1] +  ["pred_Satisfaction_final"] + columns[idx_satif+1:-1]
df_3["pred_Satisfaction_final"] = y_pred_rounded

df_3 = df_3.loc[:,columns]

df_3.to_excel("./data/final_predictions/xlsx/pred_Study_3_reviews.xlsx")
df_3.to_csv("./data/final_predictions/csv/pred_Study_3_reviews.csv")

print("Dataset with predictions saved")

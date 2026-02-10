import os

# Define a path where you HAVE write permissions
# Replace this with your actual local work directory if different
MY_WRITABLE_DIR = "/users/eleves-a/2022/adrien.bindel/ra_work/work/hf_cache"

# Create the directory if it doesn't exist
os.makedirs(MY_WRITABLE_DIR, exist_ok=True)

# Set these environment variables BEFORE importing transformers or trl
os.environ['HF_HOME'] = MY_WRITABLE_DIR
os.environ['TRANSFORMERS_CACHE'] = MY_WRITABLE_DIR
os.environ['HF_DATASETS_CACHE'] = MY_WRITABLE_DIR

# NOW import the rest
import torch
from trl import GRPOTrainer, GRPOConfig
from transformers import AutoConfig, AutoModelForCausalLM


import torch
import re
import numpy as np
from datasets import Dataset
from trl import GRPOTrainer, GRPOConfig
from transformers import AutoTokenizer, AutoModelForCausalLM

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

load_dotenv()


###########
import os

# Create a path where you DEFINITELY have write access
# Using absolute paths is safer than relative paths like "./"
absolute_path = os.path.abspath("./final_pipeline/models/ft_llama_models")
os.makedirs(absolute_path, exist_ok=True)

# Force HF to use this directory for all operations
os.environ['HF_HOME'] = absolute_path
os.environ['HF_DATASETS_CACHE'] = absolute_path
###########

device = "cuda" if torch.cuda.is_available() else "cpu"
embedding_model = SentenceTransformer( "sentence-transformers/all-mpnet-base-v2", cache_folder="embedding_models").to(device)



df_1 = pd.read_csv("./data/cleaned_data/cleaned_Study_1_reviews.csv")

## Attribute presence

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

# 1. Dataset Preparation
def prepare_grpo_dataset(df, attribute_names):
    dataset_list = []
    for _, row in df.iterrows():
        # We pick one attribute per prompt to keep the learning task focused
        for attr in attribute_names:
            if row[attr] != 0:
                prompt = (
                    f"Review: {row['finalReview']}\n"
                    f"Attribute: {attr}\n"
                    f"Predict sentiment score (1-9): <res>"
                )
                dataset_list.append({
                    "prompt": prompt,
                    "target_score": float(row['Satisfaction_final']),
                    "attribute": attr
                })
    return Dataset.from_list(dataset_list)

# 2. The Accuracy Reward Function
# This replaces the need for format rewards because we will use 
# constrained generation or simple parsing.
def accuracy_reward_func(prompts, completions, target_score, **kwargs):
    rewards = []
    for completion, target in zip(completions, target_score):
        # Extract the first digit found in the completion
        match = re.search(r'\d', completion)
        if match:
            predicted = int(match.group())
            # Reward: Inverse of the distance to the global satisfaction score
            # If they match exactly, reward is high. If they are 8 units apart, reward is 0.
            distance = abs(predicted - target)
            reward = max(0.0, 1.0 - (distance / 4.0)) 
            rewards.append(reward)
        else:
            rewards.append(-1.0) # Penalty for no digit
    return rewards

# 3. Model & Trainer Configuration
model_id = "meta-llama/Llama-3.2-3B"
custom_cache_dir = "./final_pipeline/models/ft_llama_models"
hf_token = os.getenv("HF_TOKEN")

model_id = "meta-llama/Llama-3.2-3B"
# Use the absolute path we created above
custom_cache_dir = os.path.abspath("./final_pipeline/models/ft_llama_models")

import os
import torch
from peft import LoraConfig
from trl import GRPOTrainer, GRPOConfig

# 1. PEFT (LoRA) Configuration
# This is the "magic" that makes it fit. 
# We only train a small fraction of the model's weights.
peft_config = LoraConfig(
    r=8,                       # Rank: lower = less VRAM. 8 is plenty for this task.
    lora_alpha=16,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"], # Target attention layers
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

# 2. Updated Training Configuration
training_args = GRPOConfig(
    output_dir="./llama-sentiment-rl",
    learning_rate=1e-5,        # Slightly higher LR is okay with LoRA
    per_device_train_batch_size=1,
    gradient_accumulation_steps=16, # Increase this to maintain stable training
    num_generations=4,         
    max_prompt_length=256,     # Keep this as small as possible
    max_completion_length=2,   
    bf16=True,                 # Better stability than fp16 on modern GPUs
    gradient_checkpointing=True, # DRAMATICALLY reduces VRAM by recomputing activations
    report_to="none",
    model_init_kwargs={
        "cache_dir": "/users/eleves-a/2022/adrien.bindel/ra_work/work/hf_cache",
        "token": os.getenv("HF_TOKEN"),
        "device_map": "auto",
        "torch_dtype": torch.bfloat16,
    }
)

# 3. Initialize Trainer with PEFT
trainer = GRPOTrainer(
    model=model_id,
    reward_funcs=[accuracy_reward_func],
    args=training_args,
    train_dataset=prepare_grpo_dataset(df_1, attribute_names),
    peft_config=peft_config, # Pass the LoRA config here!
)

# trainer.train()

trainer.train()


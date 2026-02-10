import torch
from transformers import pipeline
import pandas as pd
import numpy as np
import os
import json
import time
import re


def import_pipe(model_name, hf_token, cache_dir):

    pipe = pipeline(
        "text-generation",
        model=model_name,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        token=hf_token,
        do_sample=False,
        num_return_sequences=1,
        model_kwargs={"cache_dir": cache_dir}
    )

    if pipe.tokenizer.pad_token is None:
        pipe.tokenizer.pad_token = pipe.tokenizer.eos_token

    return pipe


def batch_calculate_sentiment(df, attr_name, pipe, batch_size=16):

    # Only process rows where the attribute is present (value != 0)
    mask = df[attr_name] != 0
    valid_indices = df[mask].index
    
    def data_generator():
        for idx in valid_indices:
            row = df.loc[idx]
            prompt = (
                "Task: Rate the sentiment of the specific Attribute mentioned in the Review.\n"
                "Scale: 1 (Very Negative) to 9 (Very Positive).\n\n"

                "Review: driver was superbly polite"
                "Attribute: Driver_professionalism"
                # "Global: 8"
                "Score: <res>9</res>"

                "Review: very bad. the cloths are nor ironed"
                "Attribute: cleaning_service_quality"
                # "Global: 2"
                "Score: <res>2</res>"

                f"Review: {row['finalReview']}"
                f"Attribute: {attr_name}"
                # f"Global: {row['Satisfaction_final']}"
                "Score: <res>"
            )
            yield prompt

    results = []
    outputs = pipe(
        data_generator(), 
        batch_size=batch_size, 
        max_new_tokens=5, 
        return_full_text=False,
        do_sample=True,    # Enable sampling
        temperature=0.4,   # Keep it very low so it stays focused but not "stuck"
        top_p=0.9
    )
    for out in outputs:
        response_text = out[0]["generated_text"].strip()

        print(response_text)
        
        # We look for the first digit that appears after our opening <res> tag
        # Even if the model includes </res> or other text, this finds the number.
        match = re.search(r'[1-9]', response_text)
        
        if match:
            score = int(match.group())
            results.append(score)
        else:
            # Diagnostic: what is the model actually saying when it fails?
            print(f"Extraction failed for: '{response_text}'")
            results.append(np.nan)

    df.loc[valid_indices, f"{attr_name}_sentiment"] = results
    return df



import torch
from transformers import LogitsProcessor, LogitsProcessorList

class SentimentLogitsProcessor(LogitsProcessor):
    def __init__(self, tokenizer, valid_scores=["1", "2", "3", "4", "5", "6", "7", "8", "9"]):
        # Get the token IDs for our allowed scores
        self.allowed_token_ids = [tokenizer.encode(s, add_special_tokens=False)[-1] for s in valid_scores]

    def __call__(self, input_ids, scores):
        # Create a mask of negative infinity
        mask = torch.full_like(scores, float("-inf"))
        # Only allow our specific digit tokens
        mask[:, self.allowed_token_ids] = 0
        return scores + mask

def batch_calculate_sentiment_fixed(df, attr_name, pipe, batch_size=16):
    mask = df[attr_name] != 0
    valid_indices = df[mask].index
    
    # Pre-calculate token IDs for extraction
    # We want the token IDs for "1", "2", etc.
    allowed_ids = [pipe.tokenizer.encode(str(i), add_special_tokens=False)[-1] for i in range(1, 10)]
    logits_processor = LogitsProcessorList([SentimentLogitsProcessor(pipe.tokenizer)])

    def data_generator():
        for idx in valid_indices:
            row = df.loc[idx]
            # Shortened and sharpened prompt
            prompt = (
                f"Rate {attr_name} sentiment from 1 (worst) to 9 (best).\n"
                f"Review: {row['finalReview']}\n"
                f"Score: <res>"
            )
            yield prompt

    results = []
    # Use the pipeline but with constrained generation
    outputs = pipe(
        data_generator(),
        batch_size=batch_size,
        max_new_tokens=1,  # We only need one token: the digit
        return_full_text=False,
        logits_processor=logits_processor,
        pad_token_id=pipe.tokenizer.eos_token_id
    )

    for out in outputs:
        # Since we constrained logits, this WILL be a digit 1-9
        digit = out[0]["generated_text"].strip()
        print(digit)
        try:
            # Just in case of weird whitespace/special chars
            match = re.search(r'[1-9]', digit)
            results.append(int(match.group()) if match else np.nan)
        except:
            results.append(np.nan)

    df.loc[valid_indices, f"{attr_name}_sentiment"] = results
    return df

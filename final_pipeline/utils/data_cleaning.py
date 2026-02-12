import pandas as pd
from langdetect import detect, DetectorFactory, LangDetectException

# Ensure consistent language detection results
DetectorFactory.seed = 0

def safe_detect(text):
    if not isinstance(text, str):
        return "unknown"
    
    # FIX: If text is very short, just assume English (or your main language)
    if len(text) < 15:
        return "en"
        
    try:
        return detect(text)
    except LangDetectException:
        return "unknown"

def clean_data(df, review_col):

    df[review_col] = df[review_col].fillna("").astype(str)

    df['ID'] = df.index.astype(int)
    df['ID'] += 1  # Start IDs from 1 instead of 0

    df['lang'] = df[review_col].apply(safe_detect)

    return df
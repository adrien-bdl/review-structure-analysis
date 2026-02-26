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

    # Remove rows where Review is NA, empty, or only whitespace
    df = df[df[review_col].fillna("").str.strip() != ""].copy()
    df[review_col] = df[review_col].astype(str)

    # Clean newline characters
    df.loc[:, review_col] = df[review_col].str.replace(' \n', ' ', regex=False)
    df.loc[:, review_col] = df[review_col].str.replace('\n', ' ', regex=False)

    # Add ID column
    df.loc[:, "ID"] = range(1, len(df) + 1)
    df["ID"] = df["ID"].astype(int)

    # Detect language
    df.loc[:, "lang"] = df[review_col].apply(safe_detect)

    df = df.reset_index(drop=True)

    return df

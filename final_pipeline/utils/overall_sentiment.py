from huggingface_hub import snapshot_download
from transformers import pipeline

def get_overall_sentiment(df, reviews_col_name, device, model_id, local_dir):
    snapshot_download(
        repo_id=model_id,
        local_dir=local_dir,
        local_dir_use_symlinks=False, 
        token=None
    )

    classifier = pipeline(
        "sentiment-analysis",
        model=local_dir,
        top_k=None,
        batch_size=32,
        device=device,
    )

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

    texts = df[reviews_col_name].tolist()
    raw_output = classifier(texts)

    c_scores, labels = process_results(raw_output)

    return c_scores

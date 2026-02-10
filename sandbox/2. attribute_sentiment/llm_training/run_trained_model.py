import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from peft import PeftModel
import os
from dotenv import load_dotenv

load_dotenv()

import os
from transformers import AutoModelForCausalLM
import torch

# Set environment variables to avoid /workspace issue
os.environ['HF_HOME'] = './llama_model'
os.environ['TRANSFORMERS_CACHE'] = './llama_model'

# Remove local_files_only to allow downloading
base_model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.2-3B",
    torch_dtype=torch.bfloat16,
    device_map="auto",
    token=os.getenv("HF_TOKEN"),
    # cache_dir="./llama_model",
    cache_dir="/users/eleves-a/2022/adrien.bindel/ra_work/work/llama_model",
    resume_download=True  # Resume if partially downloaded
    # Don't use local_files_only=True yet!
)

adapter_path = "./llama-sentiment-rl/checkpoint-2832" # The folder where trainer saved the results

# 2. Load the trained adapter on top of the base model
model = PeftModel.from_pretrained(base_model, adapter_path)

# 3. Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_id)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# 4. Create the inference pipeline
# The pipeline automatically handles the PEFT model
sentiment_pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)


from transformers import LogitsProcessorList

# Reuse the processor from our previous conversation
logits_processor = LogitsProcessorList([SentimentLogitsProcessor(tokenizer)])

def predict_sentiment(review, attribute):
    prompt = (
        f"Review: {review}\n"
        f"Attribute: {attribute}\n"
        f"Predict sentiment score (1-9): <res>"
    )
    
    output = sentiment_pipe(
        prompt,
        max_new_tokens=1, # Only need the score digit
        logits_processor=logits_processor,
        return_full_text=False
    )
    
    return output[0]['generated_text'].strip()

# Example Usage
score = predict_sentiment("The driver was very polite and arrived on time", "Driver_professionalism")
print(f"Predicted Score: {score}")
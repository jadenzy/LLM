import os
import re
import jieba
from transformers import AutoTokenizer, AutoModelForCausalLM

# Load the fine-tuned model and tokenizer
model_dir = "fine-tuned-llama"
tokenizer = AutoTokenizer.from_pretrained(model_dir)
model = AutoModelForCausalLM.from_pretrained(model_dir)

# Function to preprocess input text
def preprocess_text(text):
    text_no_spaces = text.replace(' ', '')  # Remove all blank spaces
    text_no_chapters = re.sub(r'第.*?章', '', text_no_spaces)  # Remove '第...章' patterns
    tokenized_text = ' '.join(jieba.cut(text_no_chapters))
    return tokenized_text

# Sample input texts for testing
input_texts = [
    "这是一个测试输入。",
    "这是另一个测试输入，带有章节信息，第1章内容。"
]

# Preprocess and tokenize the input texts
preprocessed_texts = [preprocess_text(text) for text in input_texts]
inputs = tokenizer(preprocessed_texts, return_tensors="pt", truncation=True, padding=True)

# Generate predictions
outputs = model.generate(input_ids=inputs["input_ids"], max_length=512)

# Decode the generated predictions
predictions = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]

# Display the predictions
for i, prediction in enumerate(predictions):
    print(f"Input {i+1}: {input_texts[i]}")
    print(f"Prediction {i+1}: {prediction}")
    print("\n")

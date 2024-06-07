import os
from datasets import load_dataset, Dataset
import jieba
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
import re

# Function to load and preprocess data
def load_and_preprocess_data(data_dir):
    texts = []
    pattern = re.compile(r'第.*?章')  # Regular expression pattern to match '第...章'
    
    for filename in os.listdir(data_dir):
        if filename.endswith(".txt"):
            with open(os.path.join(data_dir, filename), 'r', encoding='utf-8') as file:
                text = file.read()
                text_no_spaces = text.replace(' ', '')  # Remove all blank spaces
                text_no_chapters = re.sub(pattern, '', text_no_spaces)  # Remove '第...章' patterns
                tokenized_text = ' '.join(jieba.cut(text_no_chapters))
                texts.append(tokenized_text)
    
    if not texts:
        raise ValueError("No texts were processed. Check the data directory and file extensions.")
    
    return texts

# Load and preprocess the data
data_dir = 'data'
texts = load_and_preprocess_data(data_dir)

# Create a dataset from the concatenated texts
dataset = Dataset.from_dict({"text": texts})

# Load pre-trained model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("hfl/llama-3-chinese-8b-instruct-v3")
model = AutoModelForCausalLM.from_pretrained("hfl/llama-3-chinese-8b-instruct-v3")

# Tokenize the dataset
def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True)

tokenized_datasets = dataset.map(tokenize_function, batched=True, remove_columns=["text"])

# Set up training arguments
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=10,
)

# Initialize the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets,
)

# Start training
trainer.train()

# Save the model and tokenizer
model.save_pretrained("fine-tuned-llama")
tokenizer.save_pretrained("fine-tuned-llama")

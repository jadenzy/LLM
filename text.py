import os
from datasets import load_dataset, Dataset
import jieba
from transformers import LlamaTokenizer, LlamaForCausalLM, Trainer, TrainingArguments, DataCollatorForLanguageModeling

# Function to load and preprocess data
def load_and_preprocess_data(data_dir):
    texts = []
    for filename in os.listdir(data_dir):
        if filename.endswith(".txt"):
            with open(os.path.join(data_dir, filename), 'r', encoding='utf-8') as file:
                text = file.read()
                text_no_spaces = text.replace(' ', '')  # Remove all blank spaces
                tokenized_text = ' '.join(jieba.cut(text_no_spaces))
                texts.append(tokenized_text)
    if not texts:
        raise ValueError("No texts were processed. Check the data directory and file extensions.")
    return texts

# Load and preprocess the data
data_dir = 'data'
texts = load_and_preprocess_data(data_dir)

# Create a dataset from the concatenated texts
dataset = Dataset.from_dict({"text": texts})

# Load tokenizer and model from Hugging Face
# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("hfl/llama-3-chinese-8b-instruct-v3")
model = AutoModelForCausalLM.from_pretrained("hfl/llama-3-chinese-8b-instruct-v3")

# Add special tokens if necessary
special_tokens_dict = {'additional_special_tokens': ['[CLS]', '[SEP]']}
tokenizer.add_special_tokens(special_tokens_dict)
model.resize_token_embeddings(len(tokenizer))

# Tokenize the dataset
def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=512)

tokenized_datasets = dataset.map(tokenize_function, batched=True, num_proc=4, remove_columns=["text"])

# Define data collator
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# Define training arguments
training_args = TrainingArguments(
    output_dir="./results",
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=8,
    save_steps=10_000,
    save_total_limit=2,
    prediction_loss_only=True,
    fp16=True
)

# Initialize the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=tokenized_datasets
)

# Start training
trainer.train()

# Save the model and tokenizer
model.save_pretrained("save")
tokenizer.save_pretrained("save")

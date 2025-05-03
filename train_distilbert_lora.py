'''
Author: David Megli
Date: 2025-05-04
File: train_distilbert_lora.py
Description: Script to finetune Distilbert for Sentiment Analysis on rotten tomatoes dataset using LoRA
'''
import numpy as np
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
)
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from peft import LoraConfig, get_peft_model, TaskType

# Load dataset
dataset = load_dataset("rotten_tomatoes")

# Load tokenizer
model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Tokenize dataset
def tokenize_function(example):
    return tokenizer(example["text"], truncation=True)

tokenized_dataset = dataset.map(tokenize_function, batched=True)

# Data collator (padding in batch)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# Load base model
base_model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

# LoRA configuration
lora_config = LoraConfig(
    task_type=TaskType.SEQ_CLS,
    inference_mode=False,
    r=8,
    lora_alpha=32,
    lora_dropout=0.1,
    target_modules=["q_lin", "v_lin"]  # for DistilBERT
)

# Apply LoRA
model = get_peft_model(base_model, lora_config)
model.print_trainable_parameters()

# Compute metrics
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, predictions),
        "f1": f1_score(labels, predictions),
        "precision": precision_score(labels, predictions),
        "recall": recall_score(labels, predictions),
    }

# Training arguments
training_args = TrainingArguments(
    output_dir="./checkpoints_lora",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=2,
    logging_dir="./logs_lora",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    num_train_epochs=4,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model="f1"
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["validation"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics
)

# Train
trainer.train()

# Save model and tokenizer
trainer.save_model("./distilbert_lora_final")
tokenizer.save_pretrained("./distilbert_lora_final")

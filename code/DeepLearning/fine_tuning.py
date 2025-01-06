import torch
from transformers import LlamaForCausalLM, LlamaTokenizer, Trainer, TrainingArguments, DataCollatorForSeq2Seq
from datasets import load_dataset
from peft import LoraConfig, get_peft_model

import os

os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"

device = torch.device('mps') if torch.backends.mps.is_available() else torch.device('cpu')
print(f'Using device: {device}')

xsum_dataset = load_dataset('xsum')

print('Sample Data:', xsum_dataset['train'][0])

model_name = 'huggyllama/llama-7b'
tokenizer = LlamaTokenizer.from_pretrained(model_name, use_fast=True, legacy=False)
model = LlamaForCausalLM.from_pretrained(model_name).to(device)

tokenizer.pad_token = tokenizer.eos_token

lora_config = LoraConfig(
    r = 16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.5,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)
# model.gradient_checkpointing_enable()

# Preprocess dataset: Convert articles to inputs and summaries to labels
def preprocess_function(examples):
    inputs = ["Summarize: " + article for article in examples["document"]]
    
    # Tokenize inputs and labels
    model_inputs = tokenizer(
        inputs,
        max_length=512,
        padding="max_length",
        truncation=True,
        text_target=examples["summary"],  # Tokenize targets
    )
    
    return model_inputs

tokenized_datasets = xsum_dataset.map(preprocess_function, batched=True)

#Data Collator
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, padding=True)

#Training Arguments
training_args = TrainingArguments(
    output_dir="./llama-finetuned-xsum",
    eval_strategy="steps",
    save_strategy="steps",
    save_steps=500,
    logging_dir="./llamaft_logs",
    logging_steps=100,
    learning_rate=2e-5,
    per_device_train_batch_size=1,  # Adjust based on memory
    gradient_accumulation_steps=8,
    num_train_epochs=1,
    save_total_limit=2,
    fp16=False,  # MPS does not yet support fp16 training
    # use_cpu=True,  # Ensures MPS is used
    push_to_hub=False,
    report_to="tensorboard"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    processing_class=tokenizer,
    data_collator=data_collator
)

# Train the model
trainer.train()

# Save the model
trainer.save_model("./llama-finetuned-xsum")
tokenizer.save_pretrained("./llama-finetuned-xsum")
print("Model saved to './llama-finetuned-xsum'")


def generate_summary(prompt):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=100,
        do_sample=True,
        temperature=0.7,
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Test the fine-tuned model
test_article = "Scientists have discovered a new species of dinosaur in Argentina. The fossils indicate it was one of the largest creatures to ever walk the Earth."
summary = generate_summary("Summarize: " + test_article)
print("Generated Summary:", summary)
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
import torch
from sklearn.model_selection import train_test_split
from datasets import Dataset

# Load dataset
data = {
    "text": [
        "Check my balance",
        "What is my account balance?",
        "Show me my rewards balance",
        "Check balance",  # Ambiguous case
        "Update my phone number",
        "Change my email address",
        "Update info"  # Ambiguous case
    ],
    "intent": [
        "CheckAccountBalance",
        "CheckAccountBalance",
        "CheckRewardsBalance",
        "AmbiguousIntent",
        "UpdatePhoneNumber",
        "UpdateEmail",
        "AmbiguousIntent"
    ]
}

print("Loafing dataset")
df = pd.DataFrame(data)
# df.to_csv("intent_disambiguation_dataset.csv", index=False)

# Split the data into training and testing
train_df, test_df = train_test_split(df, test_size=0.2, random_state=45)

# Load Tokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# Preprocess data
def preprocess_data(examples):
    return tokenizer(
        examples["text"], padding=True, truncation=True
    )
    
train_dataset = Dataset.from_pandas(train_df)
test_dataset = Dataset.from_pandas(test_df)


train_dataset = train_dataset.map(preprocess_data, batched=True)
test_dataset = test_dataset.map(preprocess_data, batched=True)

# Model Setup
model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=3) # As we have 3 intents (Adjust label based on out intents)

# Trainig Arguments
training_args = TrainingArguments(
    output_dir="./new_disambiguation",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    num_train_epochs=3,
    eval_strategy="epoch"
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset
)

trainer.train()

def resolve_ambiguity(user_input):
    clarifications = {
        "Check balance": "Do you mean your account balance or your rewards balance?",
        "Update info": "What would you like to update? Your email, phone number, or address?"
    }
    return clarifications.get(user_input, None)

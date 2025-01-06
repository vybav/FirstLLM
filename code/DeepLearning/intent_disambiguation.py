from datasets import load_dataset, DatasetDict
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
import torch

# Check if MPS device is available
device = (
    torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
)
print(f"Using device: {device}")


def data_preparation():
    print("Data prep: Loading CLININC150 dataset")
    clinic_150 = load_dataset("clinc_oos", "imbalanced", split="train")
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    def preprocess_data(data):
        return tokenizer(
            data["text"], padding="max_length", truncation=True, max_length=25
        )

    tokenized_data = clinic_150.map(preprocess_data, batched=True)
    tokenized_data = tokenized_data.map(
        lambda examples: {"labels": examples["intent"]}, batched=True
    )
    return tokenized_data, tokenizer


def train_model(tokenized_data):
    model = AutoModelForSequenceClassification.from_pretrained(
        "bert-base-uncased", num_labels=150
    ).to(device)

    training_args = TrainingArguments(
        output_dir="./results",
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=5e-5,
        per_device_train_batch_size=16,
        num_train_epochs=5,
        weight_decay=0.01,
        logging_dir="./logs",
    )
    train_dataset = DatasetDict(
        {"train": tokenized_data, "eval": tokenized_data.select(range(100))}
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset["train"],
        eval_dataset=train_dataset["eval"],
    )

    trainer.train()
    return model


def predict_intent(model, tokenizer, input_text, threshold=0.7):
    # Tokenize input
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}  # Move inputs to the device

    # Get model predictions
    outputs = model(**inputs)
    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
    max_prob, pred_label = torch.max(probs, dim=-1)

    # Check confidence threshold
    if max_prob < threshold:
        return "Ambiguous Intent", max_prob.item()

    return pred_label.item(), max_prob.item()


# Define clarifying question
def clarify_intent():
    return "Can you provide more details about what you need?"


# Start a testing loop
def chatbot_interaction(model, tokenizer):
    user_input = input("You: ")  # Get user input
    if user_input.lower() == "exit":  # Allow user to exit
        print("Chatbot: Goodbye!")
        return False  # Exit condition for the loop

    # Predict the intent
    intent, confidence = predict_intent(model, tokenizer, user_input)
    if intent == "Ambiguous Intent":
        print(f"Chatbot: {clarify_intent()}")
    else:
        print(
            f"Chatbot: I understand your intent (ID: {intent}) with confidence {confidence:.2f}."
        )
    return True  # Continue the loop


def resolve_ambiguity(input_text):
    ambiguous_cases = {
        "Check balance": "Are you referring to your account balance or rewards balance?",
        "Update info": "What information would you like to update? Your email, phone number, or address?",
        "I need help with my account": "Can you clarify whether you need help updating your account details or checking your balance?",
    }

    # If input matches an ambiguous case, return the clarifying question
    for ambiguous_input, clarification in ambiguous_cases.items():
        if ambiguous_input.lower() in input_text.lower():
            return clarification

    return None


if __name__ == "__main__":
    tokenized_data, tokenizer = data_preparation()
    model = train_model(tokenized_data)
    keep_running = True

    print("Chatbot is running! Type 'exit' to stop.")
    while keep_running:
        keep_running = chatbot_interaction(model, tokenizer)

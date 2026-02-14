from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer

# hf auth login

# Load base model
model_name = "meta-llama/Llama-3.2-3B"  # adjust if using smaller version
tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=True)
model = AutoModelForCausalLM.from_pretrained(model_name, use_auth_token=True)

# LoRA config
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)

# Load dataset
dataset = load_dataset("json", data_files="dataset.jsonl")


# Tokenize
def tokenize(batch):
    return tokenizer(batch["instruction"], text_target=batch["response"], truncation=True)


tokenized_dataset = dataset.map(tokenize, batched=True)

# Training arguments
training_args = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=2,
    num_train_epochs=3,
    learning_rate=2e-4,
    fp16=True,
    logging_steps=10,
    save_steps=500
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"]
)

trainer.train()

# Save adapter weights
model.save_pretrained("./adapter_model")

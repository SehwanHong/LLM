from transformers import AutoModelForSequenceClassification, AutoTokenizer
from datasets import load_dataset

from transformers import DataCollatorWithPadding
from transformers import TrainingArguments
from transformers import Trainer

if __name__ == "__main__":
    model = AutoModelForSequenceClassification.from_pretrained("distilbert/distilbert-base-uncased", torch_dtype="auto")
    tokenizer = AutoTokenizer.from_pretrained("distilbert/distilbert-base-uncased")
    dataset = load_dataset("rotten_tomatoes")

    print("[DEBUG] Dataset loaded")
    print(dataset)

    def tokenize_dataset(dataset):
        return tokenizer(dataset["text"], padding=True, truncation=True)
    dataset = dataset.map(tokenize_dataset, batched=True)

    print("[DEBUG] Tokenized dataset")
    print(dataset)

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    print("[DEBUG] Data collator")
    print(data_collator)

    print("-" * 64)

    training_args = TrainingArguments(
        output_dir="distilbert-rotten-tomatoes",
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=2,
        push_to_hub=False,
        logging_dir="logs",
        logging_steps=10,
        save_steps=10,
        save_total_limit=2,
        local_rank=-1,
    )

    print("[DEBUG] Training arguments")
    print(training_args)

    print("-" * 64)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    trainer.train()

    print("[DEBUG] Training completed")
    print(trainer.evaluate())

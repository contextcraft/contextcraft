from transformers import RobertaForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset, DatasetDict

class CodeBERTFineTuner:
    def __init__(self, num_labels):
        self.model = RobertaForSequenceClassification.from_pretrained('microsoft/codebert-base', num_labels=num_labels)

    def fine_tune(self, train_df, test_df):
        # Convert pandas DataFrame into Hugging Face Dataset format
        dataset = DatasetDict({
            'train': Dataset.from_pandas(train_df),
            'test': Dataset.from_pandas(test_df)
        })

        training_args = TrainingArguments(
            output_dir="./results",
            evaluation_strategy="epoch",
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            num_train_epochs=3,
            weight_decay=0.01,
            logging_dir='./logs',
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=dataset['train'],
            eval_dataset=dataset['test'],
        )

        trainer.train()
        self.model.save_pretrained('./codebert-methodname')

        return self.model

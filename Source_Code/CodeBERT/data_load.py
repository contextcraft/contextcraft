import pandas as pd
from transformers import RobertaTokenizer

class DataLoad:
    def __init__(self, train_file, test_file):
        self.train_file = train_file
        self.test_file = test_file
        self.tokenizer = RobertaTokenizer.from_pretrained('microsoft/codebert-base')

    def load_data(self):
        train_df = pd.read_csv(self.train_file)
        test_df = pd.read_csv(self.test_file)
        return train_df, test_df

    def preprocess(self, dataset):
        def preprocess_function(examples):
            return self.tokenizer(examples['Functional Description'], padding="max_length", truncation=True, max_length=128)
        return dataset.map(preprocess_function, batched=True)
